import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import argparse
import accelerate
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5


parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_model_name_or_path", type=str, default="./model/gte-Qwen1.5-7B-instruct")
parser.add_argument("--clf_model_path", type=str, default="./model/TextFluoroscopy")
parser.add_argument("--use_bfloat16", type=bool, default=True)
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/text_fluoroscopy')
parser.add_argument('--save_file', type=str, default='eval_text_fluoroscopy.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')

torch.set_grad_enabled(False)

class TextFluoroscopy(nn.Module):
    def __init__(self, pretrained_model_name_or_path, clf_model_path, use_bfloat16=True, device='cuda'):
        super(TextFluoroscopy, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.embedding_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True,
                                                                    torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                                                                    device_map='auto')
        self.embedding_model = self.embedding_model.to(device=device)
        self.clf_model = AutoModel.from_pretrained(clf_model_path, trust_remote_code=True,
                                                   torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
                                                   device_map='auto')
        self.clf_model = self.clf_model.to(device=device)
        self.clf_model.eval()
        self.device = device

    def last_token_pool(self, last_hidden_states,attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device='cpu'), sequence_lengths]
        
    def get_embedding(self,input_texts):
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**batch_dict,output_hidden_states=True)
            last_logits = self.embedding_model.lm_head(outputs.hidden_states[-1]).squeeze()
            first_logits = self.embedding_model.lm_head(outputs.hidden_states[0]).squeeze()
        kls = []
        for i in range(1, len(outputs.hidden_states)-1):
            with torch.no_grad():
                middle_logits = self.embedding_model.lm_head(outputs.hidden_states[i]).squeeze()
            kls.append(F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(first_logits, dim=-1), reduction='batchmean').item()+
                       F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(last_logits, dim=-1), reduction='batchmean').item())
        max_kl_idx = kls.index(max(kls))
        max_kl_embedding = self.last_token_pool(outputs.hidden_states[max_kl_idx+1].cpu(), batch_dict['attention_mask'])
        return max_kl_embedding
    
    def forward(self, texts):
        embs = []
        for text in texts:
            emb = self.get_embedding([text])
            embs.append(emb)
        embs = torch.cat(embs, dim=0).to(device=self.device)
        with torch.no_grad():
            outputs = self.clf_model(embs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            return probs
    
    def compute_score(self, texts):
        return self.forward(texts).tolist()



if __name__ == "__main__":
    args = parser.parse_args()

    model = TextFluoroscopy(pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                            clf_model_path=args.clf_model_path,
                            use_bfloat16=args.use_bfloat16)
    
    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format, scoring_tokenizer=model.tokenizer)
    all_original_eval = []
    all_rewritten_eval = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    for idx, item in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating binoculars on {args.eval_data_path.split('/')[-1]}"):
        all_original_eval.extend(model.compute_score(item['original']))
        all_rewritten_eval.extend(model.compute_score(item['rewritten']))
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
        'method': 'text_fluoroscopy',
        'pretrained_model_name_or_path': args.pretrained_model_name_or_path,
        'clf_model_path': args.clf_model_path,
        'use_bfloat16': args.use_bfloat16,
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