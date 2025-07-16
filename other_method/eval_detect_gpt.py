# Revise from https://github.com/baoguangsheng/fast-detect-gpt
import os.path

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import tqdm
import argparse
import json
import accelerate
from torch.utils.data import DataLoader
from dataset import PerturbedDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5


parser = argparse.ArgumentParser()
parser.add_argument("--scoring_model_name", type=str, default="Qwen2-0.5B")
parser.add_argument("--cache_dir", type=str, default="./model/")
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--save_path', type=str, default='./results/FastDetectGPT')
parser.add_argument('--save_file', type=str, default='eval_fast_detect_gpt.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--seed', type=int, default=42)

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[-1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')


class DetectGPT(nn.Module):
    def __init__(self, scoring_model_name, cache_dir):
        super().__init__()
        self.scoring_model = from_pretrained(AutoModelForCausalLM, scoring_model_name,
                                             dict(torch_dtype=torch.float16), cache_dir)
        self.scoring_tokenizer = from_pretrained(AutoTokenizer, scoring_model_name,
                                                 {'padding_side': 'right', 'use_fast': False},
                                                 cache_dir)
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token

    def get_lls(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
            
        tokenized = self.scoring_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512, return_token_type_ids=False).to(self.scoring_model.device)
        labels = tokenized.input_ids
        
        with torch.no_grad():
            logits = self.scoring_model(**tokenized).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        attention_mask = tokenized.attention_mask[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape)
        
        masked_loss = loss * attention_mask
        
        sequence_lengths = attention_mask.sum(dim=1).clamp(min=1)
        sequence_loss = masked_loss.sum(dim=1) / sequence_lengths
        
        return (-sequence_loss).tolist()

    def forward(self, texts, perturbed_texts):
        """
        Compute DetectGPT score for a batch of texts and their perturbed versions.
        Args:
            texts: list of str, the texts to score
            perturbed_texts: list of list of str, perturbed versions for each text
        Returns:
            list of dict, each with 'score' for the corresponding text
        """
        lls = self.get_lls(texts)
        results = []
        if len(texts) > 0 and len(perturbed_texts) > 0 and len(perturbed_texts[0]) > 0:
            for i in range(len(texts)):
                p_lls = self.get_lls(perturbed_texts[i])
                perturbed_ll_mean = np.mean(p_lls)
                perturbed_ll_std = np.std(p_lls) if len(p_lls) > 1 else 1
                if perturbed_ll_std == 0:
                    perturbed_ll_std = 1
                score = (lls[i] - perturbed_ll_mean) / perturbed_ll_std
                results.append(score)
        return results


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    accelerator = accelerate.Accelerator()
    model = DetectGPT(args.scoring_model_name, args.cache_dir)

    dataset = PerturbedDataset(data_path=args.eval_data_path)
    data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model, data_loader = accelerator.prepare(model, data_loader)

    local_original_scores = []
    local_rewritten_scores = []

    for item in tqdm.tqdm(data_loader, desc=f"Computing DetectGPT criterion", disable=not accelerator.is_main_process):
        with torch.no_grad():
            local_original_scores.extend(model(item["original"], item["perturbed_original"]))
            local_rewritten_scores.extend(model(item["sampled"], item["perturbed_sampled"]))
    
    accelerator.wait_for_everyone()
    all_original_scores = accelerator.gather_for_metrics(torch.tensor(local_original_scores, device=accelerator.device)).cpu().tolist()
    all_rewritten_scores = accelerator.gather_for_metrics(torch.tensor(local_rewritten_scores, device=accelerator.device)).cpu().tolist()
            
    if accelerator.is_main_process:
        fpr, tpr, roc_auc = AUROC(neg_list=all_original_scores, pos_list=all_rewritten_scores)
        p, r, pr_auc = AUPR(neg_list=all_original_scores, pos_list=all_rewritten_scores)
        tpr_at_5 = TPR_at_FPR5(neg_list=all_original_scores, pos_list=all_rewritten_scores)
        original_score_mean = torch.mean(torch.tensor(all_original_scores)).item()
        original_score_std = torch.std(torch.tensor(all_original_scores)).item()
        rewritten_score_mean = torch.mean(torch.tensor(all_rewritten_scores)).item()
        rewritten_score_std = torch.std(torch.tensor(all_rewritten_scores)).item()
        print(f'Eval AUROC: {roc_auc:.4f} | Eval AUPR: {pr_auc:.4f}')
        best_mcc = 0.
        best_balanced_accuracy = 0.
        all_scores = all_original_scores + all_rewritten_scores
        for threshold in tqdm.tqdm(all_scores, desc="Finding best threshold"):
            mcc = MCC(neg_list=all_original_scores, pos_list=all_rewritten_scores, threshold=threshold)
            balanced_accuracy = Balanced_Accuracy(neg_list=all_original_scores, pos_list=all_rewritten_scores, threshold=threshold)
            if mcc > best_mcc:
                best_mcc = mcc
            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy

        print(f'Eval MCC: {best_mcc:.4f} | Eval Balanced Accuracy: {best_balanced_accuracy:.4f}')

        results = {
            'method': 'DetectGPT',
            'scoring_model_name': args.scoring_model_name,
            'dataset': args.eval_data_path,
            'eval_batch_size': args.eval_batch_size,
            'original_score_mean': original_score_mean,
            'original_score_std': original_score_std,
            'rewritten_score_mean': rewritten_score_mean,
            'rewritten_score_std': rewritten_score_std,
            'AUROC': roc_auc,
            'AUPR': pr_auc,
            'BEST_MCC': best_mcc,
            'BEST_BALANCED_ACCURACY': best_balanced_accuracy,
            'TPR_AT_FPR_5%': tpr_at_5,
            'original_scores': all_original_scores,
            'rewritten_scores': all_rewritten_scores,
        }
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        with open(os.path.join(args.save_path, args.save_file), 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False))
            print(f'Results written into {os.path.join(args.save_path, args.save_file)}')