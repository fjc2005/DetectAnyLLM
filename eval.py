import argparse
import os
import random
import time
import torch
import datetime
import json

from torch.utils.data import DataLoader
from accelerate import Accelerator
from core.model import DiscrepancyEstimator
from core.dataset import CustomDataset
from core.metrics import AUROC, AUPR, MCC, Balanced_Accuracy
from core.trainer import Trainer
from tqdm import tqdm

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--scoring_model_name', type=str, default='gpt-neo-2.7b', help='The name of the scoring model. Default: gpt-neo-2.7b.')
parser.add_argument('--reference_model_name', type=str, default=None, help='The name of the reference model. Default: None. Which indicates that the reference model is the same as the scoring model when using DPO and None when using DDL.')
parser.add_argument('--cache_dir', type=str, default='./model/', help='The directory to cache the models. Default: ./model/')
parser.add_argument('--train_method', type=str, default='DDL', help='The training method. Should be DDL or SPO. Default: DDL.')
parser.add_argument('--pretrained_model_name_or_path', type=str, default=None, help='Pass a pretrained model name or path to load the pretrained model. Default: None.')
# Dataset
parser.add_argument('--eval_data_path', type=str, default='./data/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
# WandB
parser.add_argument('--wandb', type=bool, default=False, help='Whether to use wandb for logging. Default: False.')
parser.add_argument('--wandb_dir', type=str, default='./log/', help='The directory to store the wandb logs. Default: ./log/.')
# Save
parser.add_argument('--save_dir', type=str, default='./results/', help='The directory to save the evaluation results. Default: ./results/.')
parser.add_argument('--save_file', type=str, default=None, help='The file to save the evaluation results. Default: None.')

def main(args):
    if args.wandb:
        os.makedirs(args.wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = args.wandb_dir  # 指定日志目录
    if args.save_file is not None:
        save_name = f'{args.save_file}'
    elif args.pretrained_model_name_or_path is not None:
        save_name = f'{args.pretrained_model_name_or_path.split("/")[-1]}_{args.eval_data_path.split("/")[-1].split(".json")[0]}_evalBS{args.eval_batch_size}'
    else:
        save_name = f'{args.scoring_model_name}_{args.eval_data_path.split("/")[-1].split(".json")[0]}_evalBS{args.eval_batch_size}'

    
    # Set up accelerator
    if args.wandb == True:
        import wandb
        accelerator = Accelerator(log_with='wandb')
        if args.pretrained_model_name_or_path is not None:
            accelerator.init_trackers(project_name=f'eval_{save_name[:100]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}',
                                      config={
                                          'pretrained_model_name_or_path': args.pretrained_model_name_or_path,
                                          'eval_dataset': args.eval_data_path.split("/")[-1].split(".json")[0],
                                          'eval_batch_size': args.eval_batch_size,
                                          'wandb_dir': args.wandb_dir,
                                      },
                                      init_kwargs={"wandb": {"entity": "fujiachen-nankai-university"}})
    else:
        accelerator = Accelerator()
    model = DiscrepancyEstimator(scoring_model_name=args.scoring_model_name,
                                 reference_model_name=args.reference_model_name,
                                 cache_dir=args.cache_dir,
                                 train_method=args.train_method,
                                 pretrained_ckpt=args.pretrained_model_name_or_path)
    eval_dataset = CustomDataset(args.eval_data_path,
                                 scoring_tokenizer=model.scoring_tokenizer,
                                 reference_tokenizer=model.reference_tokenizer,
                                 data_format=args.eval_data_format)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=eval_dataset.collate_fn)
    
    model, eval_dataset, eval_loader = accelerator.prepare(model, eval_dataset, eval_loader)

    # Evaluate
    trainer = Trainer()

    local_original_eval, local_rewritten_eval = trainer.eval(model, eval_loader, show_progress_bar=accelerator.is_main_process)
    accelerator.wait_for_everyone()

    all_original_eval = accelerator.gather_for_metrics(torch.tensor(local_original_eval, device=accelerator.device)).cpu().tolist()
    all_rewritten_eval = accelerator.gather_for_metrics(torch.tensor(local_rewritten_eval, device=accelerator.device)).cpu().tolist()
    if accelerator.is_main_process:
        fpr, tpr, eval_auroc = AUROC(all_original_eval, all_rewritten_eval)
        prec, recall, eval_aupr = AUPR(all_original_eval, all_rewritten_eval)
        original_discrepancy_mean = torch.mean(torch.tensor(all_original_eval)).item()
        original_discrepancy_std = torch.std(torch.tensor(all_original_eval)).item()
        rewritten_discrepancy_mean = torch.mean(torch.tensor(all_rewritten_eval)).item()
        rewritten_discrepancy_std = torch.std(torch.tensor(all_rewritten_eval)).item()
        accelerator.print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f}')
        best_mcc = 0.
        best_balanced_accuracy = 0.
        all_discrepancy = all_original_eval + all_rewritten_eval
        for threshold in tqdm(all_discrepancy, total=len(all_discrepancy), desc="Finding best threshold"):
            mcc = MCC(all_original_eval, all_rewritten_eval, threshold)
            balanced_accuracy = Balanced_Accuracy(all_original_eval, all_rewritten_eval, threshold)
            if mcc > best_mcc:
                best_mcc = mcc
            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
        accelerator.print(f'Eval MCC: {best_mcc:.4f} | Eval Balanced Accuracy: {best_balanced_accuracy:.4f}')
    
        result_dict = {
            'ckpt_name': args.pretrained_model_name_or_path if args.pretrained_model_name_or_path is not None else args.scoring_model_name,
            'eval_dataset': args.eval_data_path.split("/")[-1].split(".json")[0],
            'eval_batch_size': args.eval_batch_size,
            'original_discrepancy_mean': original_discrepancy_mean,
            'original_discrepancy_std': original_discrepancy_std,
            'rewritten_discrepancy_mean': rewritten_discrepancy_mean,
            'rewritten_discrepancy_std': rewritten_discrepancy_std,
            'AUROC': eval_auroc,
            'AUPR': eval_aupr,
            'BEST_MCC': best_mcc,
            'BEST_BALANCED_ACCURACY': best_balanced_accuracy,
            'original_discrepancy': all_original_eval,
            'rewritten_discrepancy': all_rewritten_eval,
        }
        accelerator.log(result_dict, step=0)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        with open(os.path.join(args.save_dir, f'{save_name.strip(".json")}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)