# Revise from https://github.com/baoguangsheng/fast-detect-gpt
import os
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import accelerate
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen2-0.5B")
parser.add_argument("--cache_dir", type=str, default="./model/")
parser.add_argument("--method", type=str, default="likelihood", choices=["likelihood", "rank", "logrank", "entropy"])
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/FastDetectGPT')
parser.add_argument('--save_file', type=str, default='eval_baselines.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')

torch.set_grad_enabled(False)

def get_likelihood(logits, labels, attention_mask=None):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :log_likelihood.shape[1]]
        log_likelihood = (log_likelihood * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        log_likelihood = log_likelihood.mean(dim=-1)
    return log_likelihood

def get_rank(logits, labels, attention_mask=None):
    sorted_indices = logits.argsort(dim=-1, descending=True)
    ranks = (sorted_indices == labels.unsqueeze(-1)).long().argmax(-1)
    ranks = ranks.float() + 1
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :ranks.shape[1]]
        ranks = (ranks * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        ranks = ranks.mean(dim=-1)
    return -ranks

def get_logrank(logits, labels, attention_mask=None):
    sorted_indices = logits.argsort(dim=-1, descending=True)
    ranks = (sorted_indices == labels.unsqueeze(-1)).long().argmax(-1)
    ranks = ranks.float() + 1
    log_ranks = torch.log(ranks)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :log_ranks.shape[1]]
        log_ranks = (log_ranks * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        log_ranks = log_ranks.mean(dim=-1)
    return -log_ranks

def get_entropy(logits, labels, attention_mask=None):
    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :entropy.shape[1]]
        entropy = (entropy * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        entropy = entropy.mean(dim=-1)
    return entropy

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[-1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')


methods = {
    'likelihood': get_likelihood,
    'rank': get_rank,
    'logrank': get_logrank,
    'entropy': get_entropy,
}


class Baselines(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 scoring_model: AutoModelForCausalLM=None,
                 scoring_tokenizer: AutoTokenizer=None,
                 cache_dir: str=None,
                 method: callable=get_rank,
                 ):
        super().__init__()
        self.cache_dir = cache_dir
        if scoring_model_name is not None:
            self.scoring_model_name = scoring_model_name
            self.scoring_model = from_pretrained(AutoModelForCausalLM,
                                                 scoring_model_name,
                                                 cache_dir=cache_dir,
                                                 kwargs=dict(torch_dtype=torch.float16))
            self.scoring_tokenizer = from_pretrained(AutoTokenizer,
                                                     scoring_model_name,
                                                     kwargs={'padding_side': 'right',
                                                             'use_fast': True if 'facebook/opt-' not in scoring_model_name else False},
                                                     cache_dir=cache_dir,)
        else:
            assert scoring_model is not None and scoring_tokenizer is not None, "You should provide scoring_model_name or scoring_model and scoring_tokenizer."
            self.scoring_model = scoring_model
            self.scoring_tokenizer = scoring_tokenizer
            self.scoring_model_name = scoring_model.config._name_or_path
        
        self.method = method
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token

    def forward(self, input_text):
        if isinstance(input_text, str):
            input_texts = [input_text]
        else:
            input_texts = input_text

        # original text
        tokenized = self.scoring_tokenizer(input_texts, return_tensors="pt", padding=True, return_token_type_ids=False)
        tokenized = {k: v.to(self.scoring_model.device) for k, v in tokenized.items()}
        labels = tokenized['input_ids'][:, 1:]
        attention_mask = tokenized['attention_mask'] if 'attention_mask' in tokenized else None
        if attention_mask is not None:
            attention_mask = attention_mask
        with torch.no_grad():
            logits = self.scoring_model(**tokenized).logits[:, :-1]
            scores = self.method(logits, labels, attention_mask=attention_mask)

        return scores.tolist()


if __name__ == "__main__":
    args = parser.parse_args()
    accelerator = accelerate.Accelerator()
    model = Baselines(scoring_model_name=args.scoring_model_name,
                      cache_dir=args.cache_dir,
                      method=methods[args.method])
    model.eval()
    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    local_original_eval = []
    local_rewritten_eval = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    model, data_loader = accelerator.prepare(model, data_loader)
    for idx, item in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating {args.method} on {args.eval_data_path.split('/')[-1]}"):
        local_original_eval.extend(model(item['original']))
        local_rewritten_eval.extend(model(item['rewritten']))
    accelerator.wait_for_everyone()
    all_original_eval = accelerator.gather_for_metrics(torch.tensor(local_original_eval, device=accelerator.device)).cpu().tolist()
    all_rewritten_eval = accelerator.gather_for_metrics(torch.tensor(local_rewritten_eval, device=accelerator.device)).cpu().tolist()
    if accelerator.is_main_process:
        fpr, tpr, eval_auroc = AUROC(neg_list=all_original_eval, pos_list=all_rewritten_eval)
        prec, recall, eval_aupr = AUPR(neg_list=all_original_eval, pos_list=all_rewritten_eval)
        tpr_at_5 = TPR_at_FPR5(neg_list=all_original_eval, pos_list=all_rewritten_eval)
        original_discrepancy_mean = torch.mean(torch.tensor(all_original_eval)).item()
        original_discrepancy_std = torch.std(torch.tensor(all_original_eval)).item()
        rewritten_discrepancy_mean = torch.mean(torch.tensor(all_rewritten_eval)).item()
        rewritten_discrepancy_std = torch.std(torch.tensor(all_rewritten_eval)).item()
        print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f}')
        best_mcc = 0.
        best_balanced_accuracy = 0.
        all_discrepancy = all_original_eval + all_rewritten_eval
        for threshold in tqdm(all_discrepancy, total=len(all_discrepancy), desc="Finding best threshold"):
            mcc = MCC(neg_list=all_original_eval, pos_list=all_rewritten_eval, threshold=threshold)
            balanced_accuracy = Balanced_Accuracy(neg_list=all_original_eval, pos_list=all_rewritten_eval, threshold=threshold)
            if mcc > best_mcc:
                best_mcc = mcc
            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
        print(f'Eval MCC: {best_mcc:.4f} | Eval Balanced Accuracy: {best_balanced_accuracy:.4f}')

        result_dict = {
            'method': args.method,
            'scoring_model_name': args.scoring_model_name,
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
            'TPR_AT_FPR_5%': tpr_at_5,
            'original_discrepancy': all_original_eval,
            'rewritten_discrepancy': all_rewritten_eval,
        }

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        with open(os.path.join(args.save_path, args.save_file), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))