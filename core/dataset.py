from torch.utils.data import Dataset
import json


class CustomDataset(Dataset):
    def __init__(self,
                 data_path,
                 scoring_tokenizer,
                 reference_tokenizer=None,
                 data_format='MIRAGE'):
        super().__init__()
        self.data = json.load(open(data_path, 'r'))
        self.scoring_tokenizer = scoring_tokenizer
        self.reference_tokenizer = reference_tokenizer
        self.data_format = data_format

    def __getitem__(self, index):
        if self.data_format == 'MIRAGE':
            original_text = self.data[index]['original']
            rewritten_text = self.data[index]['rewritten']
        else:
            original_text = self.data['original'][index]
            rewritten_text = self.data['rewritten'][index]
        return {
            'original': original_text,
            'rewritten': rewritten_text
        }
    
    def collate_fn(self, batch):
        original_texts = [item['original'] for item in batch]
        rewritten_texts = [item['rewritten'] for item in batch]

        # Tokenize batches with padding and truncation
        original_tokens_for_scoring_model = self.scoring_tokenizer(
            original_texts, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False
        )
        rewritten_tokens_for_scoring_model = self.scoring_tokenizer(
            rewritten_texts, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False
        )
        original_tokens_for_reference_model = self.reference_tokenizer(
            original_texts, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False
        ) if self.reference_tokenizer is not None else {'input_ids': None, 'attention_mask': None}
        rewritten_tokens_for_reference_model = self.reference_tokenizer(
            rewritten_texts, return_tensors="pt", padding=True, truncation=True, return_token_type_ids=False
        ) if self.reference_tokenizer is not None else {'input_ids': None, 'attention_mask': None}

        return {
            'scoring':{
                'original_input_ids': original_tokens_for_scoring_model['input_ids'],
                'original_attention_mask': original_tokens_for_scoring_model['attention_mask'],
                'rewritten_input_ids': rewritten_tokens_for_scoring_model['input_ids'],
                'rewritten_attention_mask': rewritten_tokens_for_scoring_model['attention_mask']
            },
            'reference':{
                'original_input_ids': original_tokens_for_reference_model['input_ids'],
                'original_attention_mask': original_tokens_for_reference_model['attention_mask'],
                'rewritten_input_ids': rewritten_tokens_for_reference_model['input_ids'],
                'rewritten_attention_mask': rewritten_tokens_for_reference_model['attention_mask']
            }
        }
    
    def __len__(self):
        return len(self.data) if self.data_format == 'MIRAGE' else len(self.data['original'])