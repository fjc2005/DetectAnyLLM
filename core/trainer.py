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
              eval_batch_size: int = 1):
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
            epoch_loss = 0.
            epoch_original_discrepancy_train = []
            epoch_rewritten_discrepancy_train = []
            model.train()
            for step, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Fine-tuning: {epoch+1} epoch"):
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
                epoch_loss += loss.item()
                for ori_discrepancy, rew_discrepancy in zip(outputs['scoring_original_discrepancy'], outputs['scoring_rewritten_discrepancy']):
                    epoch_original_discrepancy_train.append(ori_discrepancy.item())
                    epoch_rewritten_discrepancy_train.append(rew_discrepancy.item())
            epoch_loss /= len(train_loader)
            fpr, tpr, train_epoch_auroc = AUROC(epoch_original_discrepancy_train, epoch_rewritten_discrepancy_train)
            prec, recall, train_epoch_aupr = AUPR(epoch_original_discrepancy_train, epoch_rewritten_discrepancy_train)
            print(f'Epoch: {epoch + 1} | Time: {time.time() - epoch_start_time:.3f} sec')
            print(f'Train Loss: {epoch_loss:.8f}')
            print(f'Original Discrepancy Mean: {torch.mean(torch.tensor(epoch_original_discrepancy_train)).item():.2f} | Original Discrepancy Std: {torch.std(torch.tensor(epoch_original_discrepancy_train)).item():.2f}')
            print(f'Rewritten Discrepancy Mean: {torch.mean(torch.tensor(epoch_rewritten_discrepancy_train)).item():.2f} | Rewritten Discrepancy Std: {torch.std(torch.tensor(epoch_rewritten_discrepancy_train)).item():.2f}')
            print(f'Train AUROC: {train_epoch_auroc:.4f} | Train AUPR: {train_epoch_aupr:.4f}')

            if (epoch + 1) % eval_freq == 0:
                eval_start_time = time.time()
                original_discrepancy_mean, original_discrepancy_std, rewritten_discrepancy_mean, rewritten_discrepancy_std, \
                    eval_epoch_auroc, eval_epoch_aupr = self.eval(model, eval_loader)
                
                print(f'Epoch: {epoch + 1} | Eval Time: {time.time() - eval_start_time:.3f} sec')
                print(f'Original Discrepancy Mean: {original_discrepancy_mean:.2f} | Original Discrepancy Std: {original_discrepancy_std:.2f}')
                print(f'Rewritten Discrepancy Mean: {rewritten_discrepancy_mean:.2f} | Rewritten Discrepancy Std: {rewritten_discrepancy_std:.2f}')

                print(f'Eval AUROC: {eval_epoch_auroc:.4f} | Eval AUPR: {eval_epoch_aupr:.4f}')
            
            if (epoch + 1) == num_epochs and (epoch + 1) % (eval_freq) != 0:
                eval_start_time = time.time()
                eval_discrepancy_mean, eval_discrepancy_std, eval_epoch_auroc, eval_epoch_aupr = self.eval(model, eval_loader)
                print(f'Epoch: {epoch + 1} | Eval Time: {time.time() - eval_start_time:.3f} sec')
                print(f'Eval Discrepancy Mean: {eval_discrepancy_mean:.2f} | Eval Discrepancy Std: {eval_discrepancy_std:.2f}')
                print(f'Eval AUROC: {eval_epoch_auroc:.4f} | Eval AUPR: {eval_epoch_aupr:.4f}')
            
            if (epoch + 1) == num_epochs or (epoch + 1) % save_freq == 0:
                print('saving model ...')
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_directory)
            
            if (epoch + 1) == num_epochs:
                print(f'Finished Training!')
                print(f'Total Time: {time.time() - start_time:.3f} sec')

    def eval(self,
             model,
             eval_loader):
        epoch_original_discrepancy_eval = []
        epoch_rewritten_discrepancy_eval = []
        model.eval()
        with torch.no_grad():
            for step, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Evaluating"):
                outputs = model(batch['scoring']['original_input_ids'],
                    batch['scoring']['original_attention_mask'],
                    batch['scoring']['rewritten_input_ids'],
                    batch['scoring']['rewritten_attention_mask'],
                    batch['reference']['original_input_ids'],
                    batch['reference']['original_attention_mask'],
                    batch['reference']['rewritten_input_ids'],
                    batch['reference']['rewritten_attention_mask'])
                for ori_discrepancy, rew_discrepancy in zip(outputs['scoring_original_discrepancy'], outputs['scoring_rewritten_discrepancy']):
                    epoch_original_discrepancy_eval.append(ori_discrepancy.item())
                    epoch_rewritten_discrepancy_eval.append(rew_discrepancy.item())
            fpr, tpr, eval_epoch_auroc = AUROC(epoch_original_discrepancy_eval, epoch_rewritten_discrepancy_eval)
            prec, recall, eval_epoch_aupr = AUPR(epoch_original_discrepancy_eval, epoch_rewritten_discrepancy_eval)
            original_discrepancy_mean = torch.mean(torch.tensor(epoch_original_discrepancy_eval)).item()
            original_discrepancy_std = torch.std(torch.tensor(epoch_original_discrepancy_eval)).item()
            rewritten_discrepancy_mean = torch.mean(torch.tensor(epoch_rewritten_discrepancy_eval)).item()
            rewritten_discrepancy_std = torch.std(torch.tensor(epoch_rewritten_discrepancy_eval)).item()
            return original_discrepancy_mean, original_discrepancy_std, rewritten_discrepancy_mean, rewritten_discrepancy_std, eval_epoch_auroc, eval_epoch_aupr