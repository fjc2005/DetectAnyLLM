# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import argparse
import json
import accelerate
from torch.utils.data import DataLoader
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5


parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default="./results/DNA-GPT")
parser.add_argument('--save_file', type=str, default='eval_dna_gpt.json')
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD.')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')
parser.add_argument('--regen_number', type=int, default=10)
parser.add_argument('--scoring_model_name', type=str, default="Qwen2-0.5B")
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--do_top_k', action='store_true')
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--do_top_p', action='store_true')
parser.add_argument('--top_p', type=float, default=0.96)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cache_dir', type=str, default="./model/")


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


def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[-1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')

class DNAGPT(nn.Module):
    def __init__(self, scoring_model_name, cache_dir, temperature=1.0, top_p=0.96, do_top_p=False, top_k=40, do_top_k=False, regen_number=10):
        super().__init__()
        self.scoring_model_name = scoring_model_name
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.top_p = top_p
        self.do_top_p = do_top_p
        self.top_k = top_k
        self.do_top_k = do_top_k
        self.regen_number = regen_number

        self.scoring_tokenizer = from_pretrained(AutoTokenizer, scoring_model_name,
                                                 {'padding_side': 'right', 'use_fast': False},
                                                 cache_dir)
        self.scoring_model = from_pretrained(AutoModelForCausalLM, scoring_model_name,
                                             {'torch_dtype': torch.float16, 'device_map': 'auto'},
                                             cache_dir)
        self.scoring_model.eval()

        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id


    def _sample_from_model(self, texts):
        # 只从前50个token进行regen
        # Tokenize and truncate to first 50 tokens for each text
        truncated_texts = []
        for text in texts:
            tokens = self.scoring_tokenizer.encode(text, add_special_tokens=False)
            truncated_tokens = tokens[:50]
            truncated_text = self.scoring_tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            truncated_texts.append(truncated_text)
        all_encoded = self.scoring_tokenizer(truncated_texts, return_tensors="pt", padding=True).to(self.scoring_model.device)
        
        sampling_kwargs = {'temperature': self.temperature}
        if self.do_top_p:
            sampling_kwargs['top_p'] = self.top_p
        elif self.do_top_k:
            sampling_kwargs['top_k'] = self.top_k
        
        outputs = self.scoring_model.generate(**all_encoded, min_new_tokens=50, max_new_tokens=150, do_sample=True,
                                              **sampling_kwargs, pad_token_id=self.scoring_tokenizer.eos_token_id,
                                              eos_token_id=self.scoring_tokenizer.eos_token_id)
        decoded = self.scoring_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return decoded
    
    def forward(self, input_texts):
        # 1. Get log probability of original texts
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        elif isinstance(input_texts, list):
            input_texts = input_texts
        else:
            raise ValueError(f"Invalid input type: {type(input_texts)}")
        tokenized = self.scoring_tokenizer(input_texts, return_tensors="pt", padding=True).to(self.scoring_model.device)
        labels = tokenized.input_ids[:, 1:]
        attention_mask = tokenized.attention_mask
        with torch.no_grad():
            logits_score = self.scoring_model(**tokenized).logits[:, :-1]  # b, s, vocab_size
            lprobs = get_likelihood(logits_score, labels, attention_mask)

        # 2. Generate regenerated samples
        texts_to_regen = [[text for _ in range(self.regen_number)] for text in input_texts]
        regens = [self._sample_from_model(texts) for texts in texts_to_regen]

        # 3. Get log probability of regenerated samples
        lprob_regens = []
        for regen in regens:
            regen_tokenized = self.scoring_tokenizer(regen, return_tensors="pt", padding=True).to(self.scoring_model.device)
            regen_labels = regen_tokenized.input_ids[:, 1:]
            regen_attention_mask = regen_tokenized.attention_mask
            with torch.no_grad():
                regen_logits_score = self.scoring_model(**regen_tokenized).logits[:, :-1]
                lprob_regen = get_likelihood(regen_logits_score, regen_labels, regen_attention_mask).tolist()
                # print(len(lprob_regen))
            lprob_regens.append(lprob_regen)

        # 4. Compute DNA-GPT score
        lprob_regens_mean = torch.tensor(lprob_regens, device=self.scoring_model.device).mean(dim=-1, keepdim=False)
        wscore = lprobs - lprob_regens_mean

        return wscore.tolist()



if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    accelerator = accelerate.Accelerator()

    model = DNAGPT(
        scoring_model_name=args.scoring_model_name,
        cache_dir=args.cache_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        do_top_p=args.do_top_p,
        top_k=args.top_k,
        do_top_k=args.do_top_k,
        regen_number=args.regen_number
    )
    model.eval()
    
    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    
    model, data_loader = accelerator.prepare(model, data_loader)

    local_original_scores = []
    local_rewritten_scores = []
    
    for item in tqdm.tqdm(data_loader, desc=f"Computing DNA-GPT scores", total=len(data_loader)):
        with torch.no_grad():
            # In CustomDataset, machine-generated text is 'rewritten'. Here we treat it as 'sampled'
            local_original_scores.extend(model(item['original']))
            local_rewritten_scores.extend(model(item['rewritten']))
        
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
            'method': 'DNA-GPT',
            'scoring_model_name': args.scoring_model_name,
            'method_regen_number': args.regen_number,
            'method_do_top_k': args.do_top_k,
            'method_top_k': args.top_k,
            'method_do_top_p': args.do_top_p,
            'method_top_p': args.top_p,
            'method_temperature': args.temperature,
            'eval_dataset': args.eval_data_path.split("/")[-1].split(".json")[0],
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