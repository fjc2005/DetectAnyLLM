import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
import copy
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


class DiscrepancyEsitimator(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 scoring_model: AutoModelForCausalLM=None,
                 scoring_tokenizer: AutoTokenizer=None,
                 reference_model_name: str=None,
                 reference_model: AutoModelForCausalLM=None,
                 reference_tokenizer: AutoTokenizer=None,
                 cache_dir: str=None,):
        super().__init__()
        if scoring_model_name is None:
            assert scoring_model is not None and scoring_tokenizer is not None, \
                "If scoring_model_name is not provided, you should provide specific scoring_model!"
            self.scoring_model = scoring_model
            self.scoring_tokenizer = scoring_tokenizer
        else:
            assert scoring_model is None and scoring_tokenizer is None, \
                "You should not provide scoring_model_name and scoring_model/scoring_tokenizer at the same time!"
            self.scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model_name,
                                                                      device_map='auto',
                                                                      cache_dir=cache_dir)
            self.scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model_name,
                                                                   use_fast=True if 'facebook/opt-' not in scoring_model_name else False,
                                                                   padding_side='right',
                                                                   device_map='auto',
                                                                   cache_dir=cache_dir)
        if reference_model_name is None:
            if reference_model is not None:
                assert reference_tokenizer is not None, \
                    "If reference_model is provided, you should provide reference_tokenizer!"
                self.reference_model = reference_model
                self.reference_tokenizer = reference_tokenizer
            else:
                assert reference_tokenizer is None, \
                    "If reference_tokenizer is provided, you should provide reference_model!"
                self.reference_model = None
                self.reference_tokenizer = None
            
    def add_lora_config(self, lora_config: LoraConfig):
        self.lora_config = lora_config
        self.scoring_model = get_peft_model(self.scoring_model, self.lora_config)
        self.scoring_model.print_trainable_parameters()

    def save_pretrained(self, save_directory):
        """
        Save the model's state_dict to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        # torch.save(self.state_dict(), os.path.join(save_directory, "model.bin"))
        self.scoring_model.save_pretrained(os.path.join(save_directory, "scoring_model"))
        self.scoring_tokenizer.save_pretrained(os.path.join(save_directory, "scoring_model"))

    def from_pretrained(self, load_directory):
        """
        Load the model's state_dict from the specified directory.
        """
        if not os.path.exists(load_directory):
            raise ValueError(f"Directory {load_directory} does not exist.")

        self.scoring_model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(load_directory, "scoring_model"))
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "scoring_model"))
        if 'gpt-j' in self.scoring_model_name:
            self.scoring_model.half()
            if self.reference_model is not None:
                self.reference_model.half()
            self.half()

    def get_sampling_discrepancy_analytic(self, logits_ref, logits_score, labels):

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
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        
        return discrepancy, log_likelihood.sum(dim=-1)
    
    def forward(self, original_text, rewritten_text):
        # processing original text
        tokenized = self.scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:] # shape: [bsz, sentence_len]
        logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]
        if self.reference_model is not None:
            with torch.no_grad():
                tokenized_ref = self.reference_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
                assert torch.all(tokenized_ref.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(tokenized_ref.input_ids, attention_mask=tokenized_ref.attention_mask).logits[:,:-1,:]
                original_discrepancy_ref, original_logprob_ref = self.get_sampling_discrepancy_analytic(logits_ref, logits_ref, labels)
        else:
            logits_ref = logits_score
            original_discrepancy_ref, original_logprob_ref = None, None
        original_discrepancy_score, original_logprob_score = self.get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)

        # processing rewritten text
        tokenized = self.scoring_tokenizer(rewritten_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:] # shape: [bsz, sentence_len]
        logits_score = self.scoring_model(tokenized.input_ids, attention_mask=tokenized.attention_mask).logits[:,:-1,:]
        if self.reference_model is not None:
            with torch.no_grad():
                tokenized_ref = self.reference_tokenizer(rewritten_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
                assert torch.all(tokenized_ref.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(tokenized_ref.input_ids, attention_mask=tokenized_ref.attention_mask).logits[:,:-1,:]
                rewritten_discrepancy_ref, rewritten_logprob_ref = self.get_sampling_discrepancy_analytic(logits_ref, logits_ref, labels)
        else:
            logits_ref = logits_score
            rewritten_discrepancy_ref, rewritten_logprob_ref = None, None
        rewritten_discrepancy_score, rewritten_logprob_score = self.get_sampling_discrepancy_analytic(logits_ref, logits_score, labels)

        return {
            'scoring_model_original_discrepancy': original_discrepancy_score,
            'scoring_model_original_logprob': original_logprob_score,
            'scoring_model_rewritten_discrepancy': rewritten_discrepancy_score,
            'scoring_model_rewritten_logprob': rewritten_logprob_score,
            'reference_model_original_discrepancy': original_discrepancy_ref,
            'reference_model_original_logprob': original_logprob_ref,
            'reference_model_rewritten_discrepancy': rewritten_discrepancy_ref,
            'reference_model_rewritten_logprob': rewritten_logprob_ref,
        }