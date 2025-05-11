import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
import copy
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


class DiscrepancyEstimator(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 reference_model_name: str=None,
                 cache_dir: str=None,
                 train_method: str='DDL',
                 ):
        super().__init__()
        assert train_method in ['DDL', 'SPO'], 'train_method should be DDL or SPO.'
        self.train_method = train_method
        self.scoring_model_name = scoring_model_name
        self.reference_model_name = reference_model_name
        if scoring_model_name is None:
            raise ValueError('You should provide scoring_model_name.')
        else:
            self.scoring_model = AutoModelForCausalLM.from_pretrained(scoring_model_name,
                                                                    device_map='auto',
                                                                    cache_dir=cache_dir)
            self.scoring_tokenizer = AutoTokenizer.from_pretrained(scoring_model_name,
                                                                use_fast=True if 'facebook/opt-' not in scoring_model_name else False,
                                                                padding_side='right',
                                                                device_map='auto',
                                                                cache_dir=cache_dir)
        if reference_model_name is None:
            if train_method == 'DDL':
                self.reference_model = None
                self.reference_tokenizer = None
            elif train_method == 'SPO':
                self.reference_model = copy.deepcopy(self.scoring_model)
                self.reference_tokenizer = self.scoring_tokenizer
            else:
                raise ValueError('train_method should be DDL or SPO.')
        else:
            self.reference_model = AutoModelForCausalLM.from_pretrained(reference_model_name,
                                                                        device_map='auto',
                                                                        cache_dir=cache_dir)
            self.reference_tokenizer = AutoTokenizer.from_pretrained(reference_model_name,
                                                                     use_fast=True if 'facebook/opt-' not in reference_model_name else False,
                                                                     padding_side='right',
                                                                     device_map='auto',
                                                                     cache_dir=cache_dir)
            
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

    def get_sampling_discrepancy_analytic(self, reference_logits, scoring_logits, labels, attention_mask):

        if reference_logits.size(-1) != scoring_logits.size(-1):
            vocab_size = min(reference_logits.size(-1), scoring_logits.size(-1))
            reference_logits = reference_logits[:, :, :vocab_size]
            scoring_logits = scoring_logits[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == scoring_logits.ndim - 1 else labels
        lprobs_score = torch.log_softmax(scoring_logits, dim=-1)
        probs_ref = torch.softmax(reference_logits, dim=-1)
        
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

        mask = attention_mask[:, 1:].float()  # [bsz, seq_len-1], 1 for non-pad, 0 for pad
        assert mask.shape == labels.shape, "mask and labels should have the same shape."
        log_likelihood_sum = (log_likelihood * mask).sum(dim=-1)  # [bsz], sum over non-pad tokens
        mean_ref_sum = (mean_ref * mask).sum(dim=-1)  # [bsz], sum over non-pad tokens
        var_ref_sum = (var_ref * mask).sum(dim=-1)  # [bsz], sum over non-pad tokens
        discrepancy = (log_likelihood_sum - mean_ref_sum) / (var_ref_sum.sqrt() + 1e-8)  # [bsz], avoid division by zero
        
        return discrepancy, log_likelihood_sum
    
    def get_discrepancy_of_scoring_and_reference_models(self,
                                                        input_ids_for_scoring_model,
                                                        attention_mask_for_scoring_model,
                                                        input_ids_for_reference_model=None,
                                                        attention_mask_for_reference_model=None,
                                                        ) -> dict:
        labels = input_ids_for_scoring_model[:, 1:] # shape: [bsz, sentence_len]
        scoring_logits = self.scoring_model(input_ids_for_scoring_model,
                                            attention_mask=attention_mask_for_scoring_model).logits[:,:-1,:]
        if self.reference_model is not None:
            assert input_ids_for_reference_model is not None and attention_mask_for_reference_model is not None, \
                "If reference_model is provided, you should provide reference_tokenizer to dataset initialization."
            with torch.no_grad():
                # check if tokenizer is the match
                reference_labels = input_ids_for_reference_model[:, 1:] # shape: [bsz, sentence_len]
                assert torch.all(reference_labels == labels), \
                    "Tokenizer is mismatch."
                reference_logits = self.reference_model(input_ids_for_reference_model,
                                                        attention_mask=attention_mask_for_reference_model).logits[:,:-1,:]
        else:
            reference_logits = scoring_logits

        if self.reference_model is not None:
            discrepancy_ref, logprob_ref = self.get_sampling_discrepancy_analytic(reference_logits, reference_logits,
                                                                                  labels, attention_mask=attention_mask_for_reference_model)
        else:
            discrepancy_ref, logprob_ref = None, None
        discrepancy_score, logprob_score = self.get_sampling_discrepancy_analytic(reference_logits, scoring_logits,
                                                                                  labels, attention_mask=attention_mask_for_scoring_model)

        return {
            'scoring_discrepancy': discrepancy_score,
            'scoring_logprob': logprob_score,
            'reference_discrepancy': discrepancy_ref,
            'reference_logprob': logprob_ref,
        }
    
    def forward(self,
                scoring_original_input_ids,
                scoring_original_attention_mask,
                scoring_rewritten_input_ids,
                scoring_rewritten_attention_mask,
                reference_original_input_ids=None,
                reference_original_attention_mask=None,
                reference_rewritten_input_ids=None,
                reference_rewritten_attention_mask=None,
                ) -> dict:
        if self.train_method == 'SPO':
            assert reference_original_input_ids is not None and reference_original_attention_mask is not None, \
                "If train_method is SPO, you should provide reference_original_input_ids and reference_original_attention_mask."
            assert reference_rewritten_input_ids is not None and reference_rewritten_attention_mask is not None, \
                "If train_method is SPO, you should provide reference_rewritten_input_ids and reference_rewritten_attention_mask."
        elif self.train_method == 'DDL':
            assert reference_original_input_ids is None and reference_original_attention_mask is None, \
                "If train_method is DDL, you should not provide reference_original_input_ids and reference_original_attention_mask."
            assert reference_rewritten_input_ids is None and reference_rewritten_attention_mask is None, \
                "If train_method is DDL, you should not provide reference_rewritten_input_ids and reference_rewritten_attention_mask."
        else:
            raise ValueError('train_method should be DDL or SPO.')
        original_output = self.get_discrepancy_of_scoring_and_reference_models(
            input_ids_for_scoring_model=scoring_original_input_ids,
            attention_mask_for_scoring_model=scoring_original_attention_mask,
            input_ids_for_reference_model=reference_original_input_ids,
            attention_mask_for_reference_model=reference_original_attention_mask,
        )
        rewritten_output = self.get_discrepancy_of_scoring_and_reference_models(
            input_ids_for_scoring_model=scoring_rewritten_input_ids,
            attention_mask_for_scoring_model=scoring_rewritten_attention_mask,
            input_ids_for_reference_model=reference_rewritten_input_ids,
            attention_mask_for_reference_model=reference_rewritten_attention_mask,
        )
        
        return {
            'scoring_original_discrepancy': original_output['scoring_discrepancy'],
            'scoring_original_logprob': original_output['scoring_logprob'],
            'scoring_rewritten_discrepancy': rewritten_output['scoring_discrepancy'],
            'scoring_rewritten_logprob': rewritten_output['scoring_logprob'],
            'reference_original_discrepancy': original_output['reference_discrepancy'],
            'reference_original_logprob': original_output['reference_logprob'],
            'reference_rewritten_discrepancy': rewritten_output['reference_discrepancy'],
            'reference_rewritten_logprob': rewritten_output['reference_logprob'],
        }