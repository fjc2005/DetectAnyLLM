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
import copy


parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen2-0.5B")
parser.add_argument("--reference_model_name", type=str, default=None)
parser.add_argument("--cache_dir", type=str, default="./model/")
parser.add_argument("--discrepancy_analytic", type=bool, default=False)
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/FastDetectGPT')
parser.add_argument('--save_file', type=str, default='eval_fast_detect_gpt.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')

torch.set_grad_enabled(False)

def get_samples(logits, labels):
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels, attention_mask=None):
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels).squeeze(-1)

    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :log_likelihood.shape[1]]
        log_likelihood = (log_likelihood * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
    else:
        log_likelihood = log_likelihood.mean(dim=-1)
    return log_likelihood

def get_sampling_discrepancy(logits_ref, logits_score, labels, attention_mask=None):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels, attention_mask)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples, attention_mask)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x - miu_tilde) / (sigma_tilde + 1e-8)
    return discrepancy

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels, attention_mask=None):
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

    if attention_mask is not None:
        mask = attention_mask[:, 1:].float()
        mask = mask[:, :log_likelihood.shape[1]]
        log_likelihood = (log_likelihood * mask).sum(dim=-1)
        mean_ref = (mean_ref * mask).sum(dim=-1)
        var_ref = (var_ref * mask).sum(dim=-1)
    else:
        log_likelihood = log_likelihood.sum(dim=-1)
        mean_ref = mean_ref.sum(dim=-1)
        var_ref = var_ref.sum(dim=-1)

    discrepancy = (log_likelihood - mean_ref) / (var_ref.sqrt() + 1e-8)
    return discrepancy

    
def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[-1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')


class FastDetectGPT(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 reference_model_name: str=None,
                 scoring_model: AutoModelForCausalLM=None,
                 reference_model: AutoModelForCausalLM=None,
                 scoring_tokenizer: AutoTokenizer=None,
                 reference_tokenizer: AutoTokenizer=None,
                 cache_dir: str=None,
                 discrepancy_analytic: bool=False,
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
            if scoring_model is None or scoring_tokenizer is None:
                raise ValueError('You should provide scoring_model_name or scoring_model and scoring_tokenizer.')
            self.scoring_model = scoring_model
            self.scoring_tokenizer = scoring_tokenizer
            self.scoring_model_name = scoring_model.config._name_or_path
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id

        if reference_model_name is not None:
            if reference_model_name != scoring_model_name:
                self.reference_model = from_pretrained(AutoModelForCausalLM,
                                                        reference_model_name,
                                                        cache_dir=cache_dir,
                                                        kwargs=dict(torch_dtype=torch.float16))
                self.reference_tokenizer = from_pretrained(AutoTokenizer,
                                                            reference_model_name,
                                                            kwargs={'padding_side': 'right',
                                                                    'use_fast': True if 'facebook/opt-' not in reference_model_name else False},
                                                            cache_dir=cache_dir,)
                self.reference_model_name = reference_model_name
            else:
                self.reference_model = None
                self.reference_tokenizer = None
                self.reference_model_name = self.scoring_model_name
        else:
            if reference_model is None and reference_tokenizer is None:
                self.reference_model = None
                self.reference_tokenizer = None
                self.reference_model_name = self.scoring_model_name
            elif reference_model is not None and reference_tokenizer is not None:
                self.reference_model = reference_model
                self.reference_tokenizer = reference_tokenizer
                self.reference_model_name = reference_model.config._name_or_path
            else:
                raise ValueError('You should provide reference_model and reference_tokenizer at the same time.')
        self.discrepancy = get_sampling_discrepancy_analytic if discrepancy_analytic else get_sampling_discrepancy

        if self.reference_tokenizer is not None:
            if self.reference_tokenizer.pad_token is None:
                self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token
                self.reference_tokenizer.pad_token_id = self.reference_tokenizer.eos_token_id

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        if isinstance(input_text, str):
            input_texts = [input_text]
        else:
            input_texts = input_text

        with torch.no_grad():
            tokenized_score = self.scoring_tokenizer(input_texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.scoring_model.device)
            labels = tokenized_score.input_ids[:, 1:]
            logits_score = self.scoring_model(**tokenized_score).logits[:, :-1]

            if self.scoring_model_name == self.reference_model_name:
                logits_ref = logits_score
            else:
                tokenized_ref = self.reference_tokenizer(input_texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.reference_model.device)
                logits_ref = self.reference_model(**tokenized_ref).logits[:, :-1]
                
                # Adjust for sequence length mismatch
                min_seq_len = min(logits_ref.shape[1], logits_score.shape[1])
                logits_ref = logits_ref[:, :min_seq_len, :]
                logits_score = logits_score[:, :min_seq_len, :]
                labels = labels[:, :min_seq_len]

            scores = self.discrepancy(logits_ref, logits_score, labels, tokenized_score.attention_mask)
        
        return scores.tolist()
    
    def forward(self, input_text):
        return self.compute_score(input_text)


if __name__ == "__main__":
    args = parser.parse_args()
    accelerator = accelerate.Accelerator()
    model = FastDetectGPT(scoring_model_name=args.scoring_model_name,
                          reference_model_name=args.reference_model_name,
                          cache_dir=args.cache_dir,
                          discrepancy_analytic=args.discrepancy_analytic)
    
    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    local_original_eval = []
    local_rewritten_eval = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    model, data_loader = accelerator.prepare(model, data_loader)
    for idx, item in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating FastDetectGPT on {args.eval_data_path.split('/')[-1]}"):
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
            'method': 'FastDetectGPT',
            'scoring_model_name': args.scoring_model_name,
            'reference_model_name': args.reference_model_name,
            'discrepancy_analytic': args.discrepancy_analytic,
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