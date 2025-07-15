import os
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import accelerate
import json
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from dataset import CustomDataset
from metrics import AUROC, AUPR, MCC, Balanced_Accuracy, TPR_at_FPR5


parser = argparse.ArgumentParser()
parser.add_argument("--observer_model_path", type=str, default="./model/Qwen2-0.5B")
parser.add_argument("--performer_model_path", type=str, default="./model/Qwen2-0.5B")
parser.add_argument('--eval_data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--eval_data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/binoculars')
parser.add_argument('--save_file', type=str, default='eval_binoculars.json')
parser.add_argument('--eval_batch_size', type=int, default=1, help='The batch size for evaluation. Default: 1.')

torch.set_grad_enabled(False)

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)

def perplexity(encoding: transformers.BatchEncoding,
               logits: torch.Tensor):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = encoding.input_ids[..., 1:].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

    ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
           shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
    ppl = ppl.to("cpu").float().numpy()

    return ppl


def entropy(p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int):
    vocab_size = p_logits.shape[-1]
    total_tokens_available = q_logits.shape[-2]
    p_scores, q_scores = p_logits, q_logits

    p_proba = softmax_fn(p_scores).view(-1, vocab_size)

    q_scores = q_scores.view(-1, vocab_size)

    ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
    padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

    agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())

    return agg_ce


def assert_tokenizer_consistency(model_id_1, model_id_2):
    identical_tokenizers = (
            AutoTokenizer.from_pretrained(model_id_1).vocab
            == AutoTokenizer.from_pretrained(model_id_2).vocab
    )
    if not identical_tokenizers:
        raise ValueError(f"Tokenizers are not identical for {model_id_1} and {model_id_2}.")


class Binoculars(nn.Module):
    def __init__(self,
                 observer_name_or_path: str = "./model/Qwen2-0.5B",
                 performer_name_or_path: str = "./model/Qwen2-0.5B",
                 max_token_observed: int = 512,
                 mode: str = "low-fpr",
                 ) -> None:
        assert_tokenizer_consistency(observer_name_or_path, performer_name_or_path)

        self.change_mode(mode)
        self.observer_model = AutoModelForCausalLM.from_pretrained(observer_name_or_path,
                                                                   trust_remote_code=True,
                                                                   torch_dtype=torch.float16
                                                                   )
        self.performer_model = AutoModelForCausalLM.from_pretrained(performer_name_or_path,
                                                                    trust_remote_code=True,
                                                                    torch_dtype=torch.float16
                                                                    )
        self.observer_model.eval()
        self.performer_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name_or_path)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings).logits
        performer_logits = self.performer_model(**encodings).logits
        return observer_logits, performer_logits

    def forward(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)
        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits, performer_logits,
                        encodings, self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        return binoculars_scores.tolist()



if __name__ == "__main__":
    args = parser.parse_args()
    accelerator = accelerate.Accelerator()
    model = Binoculars(observer_name_or_path=args.observer_model_path,
                       performer_name_or_path=args.performer_model_path)
    dataset = CustomDataset(data_path=args.eval_data_path, data_format=args.eval_data_format)
    local_original_eval = []
    local_rewritten_eval = []
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    model, data_loader = accelerator.prepare(model, data_loader)
    for idx, item in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating binoculars on {args.eval_data_path.split('/')[-1]}"):
        local_original_eval.extend(model(item['original']))
        local_rewritten_eval.extend(model(item['rewritten']))
    if accelerator.is_main_process:
        all_original_eval = accelerator.gather_for_metrics(torch.tensor(local_original_eval, device=accelerator.device)).cpu().tolist()
        all_rewritten_eval = accelerator.gather_for_metrics(torch.tensor(local_rewritten_eval, device=accelerator.device)).cpu().tolist()
        fpr, tpr, eval_auroc = AUROC(pos_list=all_original_eval, neg_list=all_rewritten_eval)
        prec, recall, eval_aupr = AUPR(pos_list=all_original_eval, neg_list=all_rewritten_eval)
        tpr_at_5 = TPR_at_FPR5(pos_list=all_original_eval, neg_list=all_rewritten_eval)
        original_discrepancy_mean = torch.mean(torch.tensor(all_original_eval)).item()
        original_discrepancy_std = torch.std(torch.tensor(all_original_eval)).item()
        rewritten_discrepancy_mean = torch.mean(torch.tensor(all_rewritten_eval)).item()
        rewritten_discrepancy_std = torch.std(torch.tensor(all_rewritten_eval)).item()
        print(f'Eval AUROC: {eval_auroc:.4f} | Eval AUPR: {eval_aupr:.4f}')
        best_mcc = 0.
        best_balanced_accuracy = 0.
        all_discrepancy = all_original_eval + all_rewritten_eval
        for threshold in tqdm(all_discrepancy, total=len(all_discrepancy), desc="Finding best threshold"):
            mcc = MCC(pos_list=all_original_eval, neg_list=all_rewritten_eval, threshold=threshold)
            balanced_accuracy = Balanced_Accuracy(pos_list=all_original_eval, neg_list=all_rewritten_eval, threshold=threshold)
            if mcc > best_mcc:
                best_mcc = mcc
            if balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = balanced_accuracy
        print(f'Eval MCC: {best_mcc:.4f} | Eval Balanced Accuracy: {best_balanced_accuracy:.4f}')

        result_dict = {
            'method': 'binoculars',
            'observer_model_path': args.observer_model_path,
            'performer_model_path': args.performer_model_path,
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