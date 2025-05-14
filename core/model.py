import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
import copy
from peft import get_peft_model, LoraConfig, TaskType, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[-1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')


class DiscrepancyEstimator(nn.Module):
    def __init__(self,
                 scoring_model_name: str=None,
                 reference_model_name: str=None,
                 scoring_model: AutoModelForCausalLM=None,
                 reference_model: AutoModelForCausalLM=None,
                 scoring_tokenizer: AutoTokenizer=None,
                 reference_tokenizer: AutoTokenizer=None,
                 cache_dir: str=None,
                 train_method: str='DDL',
                 pretrained_ckpt: str=None,
                 ):
        super().__init__()
        assert train_method in ['DDL', 'SPO'], 'train_method should be DDL or SPO.'
        self.train_method = train_method
        self.cache_dir = cache_dir
        if pretrained_ckpt is not None:
            self.load_pretrained(pretrained_ckpt)
        else:
            if scoring_model_name is not None:
                self.scoring_model_name = scoring_model_name
                self.scoring_model = from_pretrained(AutoModelForCausalLM,
                                                     scoring_model_name,
                                                     cache_dir=cache_dir,
                                                     kwargs={})
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
                self.reference_model = from_pretrained(AutoModelForCausalLM,
                                                       reference_model_name,
                                                       cache_dir=cache_dir,
                                                       kwargs={})
                self.reference_tokenizer = from_pretrained(AutoTokenizer,
                                                           reference_model_name,
                                                           kwargs={'padding_side': 'right',
                                                                   'use_fast': True if 'facebook/opt-' not in reference_model_name else False},
                                                           cache_dir=cache_dir,)
                self.reference_model_name = reference_model_name
            else:
                if reference_model is None and reference_tokenizer is None:
                    if train_method == 'DDL':
                        self.reference_model = None
                        self.reference_tokenizer = None
                        self.reference_model_name = None
                    else:
                        self.reference_model = copy.deepcopy(self.scoring_model)
                        self.reference_tokenizer = self.scoring_tokenizer
                        self.reference_model_name = self.reference_model.config._name_or_path
                elif reference_model is not None and reference_tokenizer is not None:
                    self.reference_model = reference_model
                    self.reference_tokenizer = reference_tokenizer
                    self.reference_model_name = reference_model.config._name_or_path
                else:
                    raise ValueError('You should provide reference_model and reference_tokenizer at the same time.')

            if self.reference_tokenizer is not None:
                if self.reference_tokenizer.pad_token is None:
                    self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token
                    self.reference_tokenizer.pad_token_id = self.reference_tokenizer.eos_token_id
            
    def add_lora_config(self, lora_config: LoraConfig):
        self.lora_config = lora_config
        self.scoring_model = get_peft_model(self.scoring_model, self.lora_config)

    def save_pretrained(self, save_directory):
        """
        Save the model's state_dict to the specified directory.
        """
        os.makedirs(save_directory, exist_ok=True)
        # torch.save(self.state_dict(), os.path.join(save_directory, "model.bin"))
        self.scoring_model.save_pretrained(os.path.join(save_directory, "scoring_model"))
        self.scoring_tokenizer.save_pretrained(os.path.join(save_directory, "scoring_model"))
        if self.reference_model is not None:
            self.reference_model.save_pretrained(os.path.join(save_directory, "reference_model"))
            self.reference_tokenizer.save_pretrained(os.path.join(save_directory, "reference_model"))

    def load_pretrained(self, load_directory):
        """
        Load the model's state_dict from the specified directory.
        """
        if not os.path.exists(load_directory):
            raise ValueError(f"Directory {load_directory} does not exist.")

        self.scoring_model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(load_directory, "scoring_model"))
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "scoring_model"))
        self.scoring_model_name = self.scoring_model.config._name_or_path

        if os.path.exists(os.path.join(load_directory, "reference_model")):
            self.reference_model = AutoModelForCausalLM.from_pretrained(os.path.join(load_directory, "reference_model"))
            self.reference_tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "reference_model"))
            self.reference_model_name = self.reference_model.config._name_or_path
        else:
            self.reference_model = None
            self.reference_tokenizer = None
            self.reference_model_name = None

        if self.scoring_tokenizer.pad_token is None:
            self.scoring_tokenizer.pad_token = self.scoring_tokenizer.eos_token
            self.scoring_tokenizer.pad_token_id = self.scoring_tokenizer.eos_token_id
        if self.reference_tokenizer is not None:
            if self.reference_tokenizer.pad_token is None:
                self.reference_tokenizer.pad_token = self.reference_tokenizer.eos_token
                self.reference_tokenizer.pad_token_id = self.reference_tokenizer.eos_token_id

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
        labels = input_ids_for_scoring_model[:, 1:] # shape: [bsz, sentence_len - 1]
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