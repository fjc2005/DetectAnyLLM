import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import datetime

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from core.model import DiscrepancyEstimator
from core.dataset import CustomDataset
from core.loss import calculate_DPO_loss, calculate_DDL_loss
from core.metrics import AUROC, AUPR
from core.trainer import Trainer
from accelerate import Accelerator
from peft import LoraConfig, TaskType

parser = argparse.ArgumentParser()
# Model
parser.add_argument('--scoring_model_name', type=str, default='gpt-neo-2.7b', help='The name of the scoring model. Default: gpt-neo-2.7b.')
parser.add_argument('--reference_model_name', type=str, default=None, help='The name of the reference model. Default: None. Which indicates that the reference model is the same as the scoring model when using DPO and None when using DDL.')
parser.add_argument('--cache_dir', type=str, default='./model/', help='The directory to cache the models. Default: ./model/')
parser.add_argument('--train_method', type=str, default='DDL', help='The training method. Should be DDL or SPO. Default: DDL.')
# LoRA
parser.add_argument('--lora_rank', type=int, default=8, help='The rank of LoRA. Default: 8.')
parser.add_argument('--lora_alpha', type=float, default=32., help='The alpha of LoRA. Default: 32.')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='The dropout of LoRA. Default: 0.1.')
# Data
parser.add_argument('--train_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/polish.json', help='The path to the training data. Default: ./data/DIG/polish.json.')
parser.add_argument('--train_data_format', type=str, default='MIRAGE', help='The format of the training data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--train_batch_size', type=int, default=1, help='The batch size of training data. Default: 1.')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size of evaluation data. Default: 1.')
# Training
parser.add_argument('--learning_rate', type=float, default=1e-4, help='The learning rate. Default: 1e-4.')
parser.add_argument('--num_epochs', type=int, default=5, help='The number of epochs. Default: 5.')
parser.add_argument('--eval_freq', type=int, default=1, help='The frequency of evaluation. Default: 1.')
parser.add_argument('--save_freq', type=int, default=5, help='The frequency of saving the model. Default: 5.')
parser.add_argument('--save_directory', type=str, default='./ckpt/', help='The directory to save the model. Default: ./ckpt/.')
parser.add_argument('--wandb', type=bool, default=False, help='Whether to use wandb for tracking. Default: False.')
# Loss
parser.add_argument('--DDL_target_original_crit', type=float, default=0., help='The target crit of original text when using DDL. Default: 0.')
parser.add_argument('--DDL_target_rewritten_crit', type=float, default=100., help='The target crit of rewritten text when using DDL. Default: 1.')
parser.add_argument('--DPO_beta', type=float, default=0.05, help='The beta of DPO. Default: 0.05.')
# Save
parser.add_argument('--ckpt_name', type=str, default=None, help='The name of the saved model. Default: None.')
parser.add_argument('--wandb_dir', type=str, default='./log/', help='The directory to save the wandb logs. Default: ./log/.')


def main(args):
    # Set up model
    model = DiscrepancyEstimator(scoring_model_name=args.scoring_model_name,
                                 reference_model_name=args.reference_model_name,
                                 cache_dir=args.cache_dir,
                                 train_method=args.train_method)
    lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
    model.add_lora_config(lora_config)

    if args.wandb:
        os.makedirs(args.wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = args.wandb_dir  # 关键修改：指定日志目录
    if args.ckpt_name is not None:
        run_name = f'{args.ckpt_name}'
    elif args.train_method == 'DDL':
        run_name = f'DDL_score_{model.scoring_model.config._name_or_path.split("/")[-1]}_ref_{"None" if model.reference_model_name is None else model.reference_model_name.split("/")[-1]}_{args.train_data_path.split("/")[-1].split(".json")[0]}_lr{args.learning_rate}_bs{args.train_batch_size}_rewTgt{args.DDL_target_rewritten_crit}_oriTgt{args.DDL_target_original_crit}_r{args.lora_rank}'
    else:
        run_name = f'SPO_score_{model.scoring_model.config._name_or_path.split("/")[-1]}_ref_{"None" if model.reference_model_name is None else model.reference_model_name.split("/")[-1]}_{args.train_data_path.split("/")[-1].split(".json")[0]}_lr{args.learning_rate}_bs{args.train_batch_size}_beta{args.DPO_beta}_r{args.lora_rank}'
    
    # Set up accelerator
    if args.wandb == True:
        import wandb
        accelerator = Accelerator(log_with='wandb')
        accelerator.init_trackers(project_name=f'Train_Machine_Generate_Text_Detection',
                                  config={
                                      'scoring_model_name': model.scoring_model_name,
                                      'reference_model_name': model.reference_model_name if model.reference_model_name is not None else 'None',
                                      'train_method': args.train_method,
                                      'train_data': args.train_data_path,
                                      'eval_data': args.eval_data_path,
                                      'lora_rank': args.lora_rank,
                                      'lora_alpha': args.lora_alpha,
                                      'lora_dropout': args.lora_dropout,
                                      'learning_rate': args.learning_rate,
                                      'num_epochs': args.num_epochs,
                                      'train_batch_size': args.train_batch_size,
                                      'eval_batch_size': args.eval_batch_size,
                                      'DDL_target_original_crit': args.DDL_target_original_crit,
                                      'DDL_target_rewritten_crit': args.DDL_target_rewritten_crit,
                                      'DPO_beta': args.DPO_beta,
                                      'ckpt_name': args.ckpt_name,
                                      'wandb_dir': args.wandb_dir,
                                      'eval_freq': args.eval_freq,
                                  },
                                  init_kwargs={"wandb": {"entity": "fujiachen-nankai-university",
                                                         "name": f'{run_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}'}})
    else:
        accelerator = Accelerator()
    if accelerator.is_main_process:
        accelerator.print(args)

    if accelerator.is_main_process:
        model.scoring_model.print_trainable_parameters()

    # Set up dataset
    train_dataset = CustomDataset(data_path=args.train_data_path,
                                  scoring_tokenizer=model.scoring_tokenizer,
                                  reference_tokenizer=model.reference_tokenizer,
                                  data_format=args.train_data_format)
    eval_dataset = CustomDataset(data_path=args.eval_data_path,
                                 scoring_tokenizer=model.scoring_tokenizer,
                                 reference_tokenizer=None,
                                 data_format=args.eval_data_format)

    # Set up loss function
    assert args.train_method in ['DDL', 'SPO'], "Invalid loss function"
    if args.train_method == 'DDL':
        loss_fn = calculate_DDL_loss
    else:
        loss_fn = calculate_DPO_loss

    # Set up trainer
    trainer = Trainer()

    trainer.train(accelerator=accelerator,
                  model=model,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  train_batch_size=args.train_batch_size,
                  eval_batch_size=args.eval_batch_size,
                  loss_fn=loss_fn,
                  learning_rate=args.learning_rate,
                  num_epochs=args.num_epochs,
                  eval_freq=args.eval_freq,
                  save_freq=args.save_freq,
                  save_directory=args.save_directory,
                  DDL_target_original_crit=args.DDL_target_original_crit,
                  DDL_target_rewritten_crit=args.DDL_target_rewritten_crit,
                  DPO_beta=args.DPO_beta,
                  track_with_wandb=args.wandb,
                  save_name=run_name)
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)