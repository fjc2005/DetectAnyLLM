# Revise from https://github.com/baoguangsheng/fast-detect-gpt
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import json
from tqdm import tqdm
import accelerate
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="roberta-base-openai-detector")
parser.add_argument("--cache_dir", type=str, default="./model/")
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD.')
parser.add_argument('--save_path', type=str, default='./results/roberta')
parser.add_argument('--save_file', type=str, default='eval_roberta.json')
parser.add_argument('--eval_batch_size', type=int, default=8, help='The batch size for evaluation.')
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


class RobertaDetector(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 scoring_model: AutoModelForSequenceClassification=None,
                 scoring_tokenizer: AutoTokenizer=None,
                 cache_dir: str=None,
                 ):
        super().__init__()
        self.cache_dir = cache_dir
        if scoring_model_name is not None:
            self.scoring_model_name = scoring_model_name
            self.scoring_model = from_pretrained(AutoModelForSequenceClassification,
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
        
        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            
    def forward(self, input_text: list[str]):
        tokenized = self.scoring_tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.scoring_model.device)
        with torch.no_grad():
            logits = self.scoring_model(**tokenized).logits
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, 0]
        return scores.tolist()


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    accelerator = accelerate.Accelerator()
    model = RobertaDetector(scoring_model_name=args.model_name, cache_dir=args.cache_dir)
    model.eval()

    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    
    model, data_loader = accelerator.prepare(model, data_loader)

    local_original_scores = []
    local_rewritten_scores = []

    for idx, item in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating {args.model_name} on {args.eval_data_path.split('/')[-1]}"):
        local_original_scores.extend(model(item['original']))
        local_rewritten_scores.extend(model(item['rewritten']))

    accelerator.wait_for_everyone()
    all_original_scores = accelerator.gather_for_metrics(torch.tensor(local_original_scores, device=accelerator.device)).cpu().tolist()
    all_rewritten_scores = accelerator.gather_for_metrics(torch.tensor(local_rewritten_scores, device=accelerator.device)).cpu().tolist()

    if accelerator.is_main_process:
        fpr, tpr, eval_auroc = AUROC(neg_list=all_original_scores, pos_list=all_rewritten_scores)
        prec, recall, eval_aupr = AUPR(neg_list=all_original_scores, pos_list=all_rewritten_scores)
        tpr_at_5 = TPR_at_FPR5(neg_list=all_original_scores, pos_list=all_rewritten_scores)

        original_score_mean = torch.mean(torch.tensor(all_original_scores)).item()
        original_score_std = torch.std(torch.tensor(all_original_scores)).item()
        rewritten_score_mean = torch.mean(torch.tensor(all_rewritten_scores)).item()
        rewritten_score_std = torch.std(torch.tensor(all_rewritten_scores)).item()
        
        print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f}')

        best_mcc = 0.
        best_balanced_accuracy = 0.
        all_scores = all_original_scores + all_rewritten_scores
        for threshold in tqdm(all_scores, total=len(all_scores), desc="Finding best threshold"):
            mcc = MCC(neg_list=all_original_scores, pos_list=all_rewritten_scores, threshold=threshold)
            balanced_accuracy = Balanced_Accuracy(neg_list=all_original_scores, pos_list=all_rewritten_scores, threshold=threshold)
            if mcc > best_mcc:
                best_mcc = mcc
            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
        
        print(f'Eval MCC: {best_mcc:.4f} | Eval Balanced Accuracy: {best_balanced_accuracy:.4f}')

        result_dict = {
            'method': 'roberta',
            'model_name': args.model_name,
            'eval_dataset': args.eval_data_path.split("/")[-1].split(".json")[0],
            'eval_batch_size': args.eval_batch_size,
            'original_score_mean': original_score_mean,
            'original_score_std': original_score_std,
            'rewritten_score_mean': rewritten_score_mean,
            'rewritten_score_std': rewritten_score_std,
            'AUROC': eval_auroc,
            'AUPR': eval_aupr,
            'BEST_MCC': best_mcc,
            'BEST_BALANCED_ACCURACY': best_balanced_accuracy,
            'TPR_AT_FPR_5%': tpr_at_5,
            'original_scores': all_original_scores,
            'rewritten_scores': all_rewritten_scores,
        }

        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        with open(os.path.join(args.save_path, args.save_file), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, ensure_ascii=False, indent=4))