import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os

from torch.utils.data import DataLoader
from .model import DiscrepancyEstimator
from .dataset import CustomDataset
from .loss import calculate_DPO_loss, calculate_DDL_loss
from .metrics import AUROC, AUPR
from accelerate import Accelerator
from tqdm import tqdm

class Trainer():
    def train(self,
              accelerator: Accelerator,
              model: DiscrepancyEstimator,
              train_dataset: CustomDataset,
              eval_dataset: CustomDataset,
              loss_fn = calculate_DDL_loss,
              learning_rate: float = 1e-4,
              num_epochs: int = 5,
              eval_freq: int = 1,
              save_freq: int = 5,
              save_directory: str='./ckpt/',
              DDL_target_original_crit: float = 0.,
              DDL_target_rewritten_crit: float = 100.,
              DPO_beta: float = 0.05,
              train_batch_size: int = 1,
              eval_batch_size: int = 1,
              track_with_wandb: bool = True,):
        assert loss_fn in [calculate_DDL_loss, calculate_DPO_loss], "Invalid loss function"
        start_time = time.time()
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                  collate_fn=train_dataset.collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False,
                                 collate_fn=eval_dataset.collate_fn)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=num_epochs * len(train_loader),
                                                            eta_min=0,
                                                            last_epoch=-1)
        model, optimizer, train_loader, eval_loader, loss_fn, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, eval_loader, loss_fn, lr_scheduler)
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_losses = []
            epoch_original_discrepancy_train = []
            epoch_rewritten_discrepancy_train = []
            model.train()
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Fine-tuning: {epoch+1} epoch", disable=not accelerator.is_local_main_process):
                outputs = model(batch['scoring']['original_input_ids'],
                                batch['scoring']['original_attention_mask'],
                                batch['scoring']['rewritten_input_ids'],
                                batch['scoring']['rewritten_attention_mask'],
                                batch['reference']['original_input_ids'],
                                batch['reference']['original_attention_mask'],
                                batch['reference']['rewritten_input_ids'],
                                batch['reference']['rewritten_attention_mask'])
                if loss_fn == calculate_DPO_loss:
                    loss, _, _, _, _ = loss_fn(
                        outputs['scoring_rewritten_logprob'],
                        outputs['scoring_original_logprob'],
                        outputs['reference_rewritten_logprob'],
                        outputs['reference_original_logprob'],
                        beta=DPO_beta)
                else:
                    loss = loss_fn(outputs['scoring_original_discrepancy'],
                                   outputs['scoring_rewritten_discrepancy'],
                                   target_original_crit=DDL_target_original_crit,
                                   target_rewritten_crit=DDL_target_rewritten_crit)
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                train_losses.append(loss.detach().cpu().item())
                epoch_original_discrepancy_train.extend(outputs['scoring_original_discrepancy'].detach().cpu().tolist())
                epoch_rewritten_discrepancy_train.extend(outputs['scoring_rewritten_discrepancy'].detach().cpu().tolist())
                accelerator.log({
                    "train/loss": loss.detach().cpu().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }, step=step + epoch * len(train_loader))
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                all_train_losses = accelerator.gather_for_metrics(torch.tensor(train_losses))
                epoch_loss = all_train_losses.mean().item()
                all_original_train = accelerator.gather_for_metrics(torch.tensor(epoch_original_discrepancy_train))
                all_rewritten_train = accelerator.gather_for_metrics(torch.tensor(epoch_rewritten_discrepancy_train))
                fpr, tpr, train_epoch_auroc = AUROC(all_original_train.cpu().tolist(), all_rewritten_train.cpu().tolist())
                prec, recall, train_epoch_aupr = AUPR(all_original_train.cpu().tolist(), all_rewritten_train.cpu().tolist())
                original_discrepancy_mean = torch.mean(all_original_train).item()
                original_discrepancy_std = torch.std(all_original_train).item()
                rewritten_discrepancy_mean = torch.mean(all_rewritten_train).item()
                rewritten_discrepancy_std = torch.std(all_rewritten_train).item()

                # Prepare log dictionary
                log_dict = {
                    "train/auroc": train_epoch_auroc,
                    "train/aupr": train_epoch_aupr,
                    "train/original_discrepancy_mean": original_discrepancy_mean,
                    "train/original_discrepancy_std": original_discrepancy_std,
                    "train/rewritten_discrepancy_mean": rewritten_discrepancy_mean,
                    "train/rewritten_discrepancy_std": rewritten_discrepancy_std,
                }

                accelerator.print(f'Epoch: {epoch + 1} | Time: {time.time() - epoch_start_time:.3f} sec')
                accelerator.print(f'Train Loss: {epoch_loss:.8f}')
                accelerator.print(f'Original Discrepancy: {original_discrepancy_mean:.2f} ± {original_discrepancy_std:.2f} | Rewritten Discrepancy: {rewritten_discrepancy_mean:.2f} ± {rewritten_discrepancy_std:.2f}')
                accelerator.print(f'Train AUROC: {train_epoch_auroc:.4f} | Train AUPR: {train_epoch_aupr:.4f}')

            if (epoch + 1) % eval_freq == 0 or (epoch + 1) == num_epochs:
                original_discrepancy_mean, original_discrepancy_std, rewritten_discrepancy_mean, rewritten_discrepancy_std, \
                    eval_epoch_auroc, eval_epoch_aupr = self.eval(accelerator, model, eval_loader)
                log_dict.update({
                    "eval/auroc": eval_epoch_auroc,
                    "eval/aupr": eval_epoch_aupr,
                    "eval/original_discrepancy_mean": original_discrepancy_mean,
                    "eval/original_discrepancy_std": original_discrepancy_std,
                    "eval/rewritten_discrepancy_mean": rewritten_discrepancy_mean,
                    "eval/rewritten_discrepancy_std": rewritten_discrepancy_std,
                })

            if track_with_wandb:
                accelerator.log(log_dict, step=(epoch + 1) * len(train_loader))

            if (epoch + 1) == num_epochs or (epoch + 1) % save_freq == 0:
                accelerator.print('saving model ...')
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_directory)
            
            if (epoch + 1) == num_epochs:
                if track_with_wandb:
                    accelerator.end_training()
                accelerator.print(f'Finished Training!')
                accelerator.print(f'Total Time: {time.time() - start_time:.3f} sec')

    def eval(self,
             accelerator,
             model,
             eval_loader,
             return_detail_discrepancy: bool = False):
        epoch_original_discrepancy_eval = []
        epoch_rewritten_discrepancy_eval = []
        model.eval()
        with torch.no_grad():
            for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Evaluating", disable=not accelerator.is_local_main_process):
                outputs = model(
                    batch['scoring']['original_input_ids'],
                    batch['scoring']['original_attention_mask'],
                    batch['scoring']['rewritten_input_ids'],
                    batch['scoring']['rewritten_attention_mask'],
                    batch['reference']['original_input_ids'],
                    batch['reference']['original_attention_mask'],
                    batch['reference']['rewritten_input_ids'],
                    batch['reference']['rewritten_attention_mask']
                )
                epoch_original_discrepancy_eval.extend(outputs['scoring_original_discrepancy'].cpu().tolist())
                epoch_rewritten_discrepancy_eval.extend(outputs['scoring_rewritten_discrepancy'].cpu().tolist())
        
        # Gather all evaluation data across processes
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            all_original_eval = accelerator.gather_for_metrics(torch.tensor(epoch_original_discrepancy_eval))
            all_rewritten_eval = accelerator.gather_for_metrics(torch.tensor(epoch_rewritten_discrepancy_eval))
            
            # Calculate evaluation metrics
            fpr, tpr, eval_epoch_auroc = AUROC(all_original_eval.cpu().tolist(), all_rewritten_eval.cpu().tolist())
            prec, recall, eval_epoch_aupr = AUPR(all_original_eval.cpu().tolist(), all_rewritten_eval.cpu().tolist())
            original_discrepancy_mean = torch.mean(all_original_eval).item()
            original_discrepancy_std = torch.std(all_original_eval).item()
            rewritten_discrepancy_mean = torch.mean(all_rewritten_eval).item()
            rewritten_discrepancy_std = torch.std(all_rewritten_eval).item()
            
            accelerator.print(f'Original Discrepancy: {original_discrepancy_mean:.2f} ± {original_discrepancy_std:.2f} | Rewritten Discrepancy: {rewritten_discrepancy_mean:.2f} ± {rewritten_discrepancy_std:.2f}')
            accelerator.print(f'Eval AUROC: {eval_epoch_auroc:.4f} | Eval AUPR: {eval_epoch_aupr:.4f}')
        
        if return_detail_discrepancy:
            return original_discrepancy_mean, original_discrepancy_std, rewritten_discrepancy_mean, rewritten_discrepancy_std, eval_epoch_auroc, eval_epoch_aupr, all_original_eval, all_rewritten_eval
        return original_discrepancy_mean, original_discrepancy_std, rewritten_discrepancy_mean, rewritten_discrepancy_std, eval_epoch_auroc, eval_epoch_aupr