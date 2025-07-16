from torch.utils.data import Dataset
import json


class CustomDataset(Dataset):
    def __init__(self,
                 data_path,
                 data_format='MIRAGE'):
        super().__init__()
        self.data = json.load(open(data_path, 'r'))
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

        return {
            'original': original_texts,
            'rewritten': rewritten_texts
        }
    
    def __len__(self):
        return len(self.data) if self.data_format == 'MIRAGE' else len(self.data['original'])


class PerturbedDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        raw_data = json.load(open(data_path, 'r'))
        # data is stored in the 'data' key, matching generate_perturbs.py output
        self.data = raw_data['data']

    def __getitem__(self, index):
        item = self.data[index]
        # Adapt keys from generate_perturbs.py ('rewritten') to what eval_detect_gpt.py expects ('sampled')
        return {
            'original': item['original'],
            'sampled': item['rewritten'],
            'perturbed_original': item['perturbed_original'],
            'perturbed_sampled': item['perturbed_rewritten'],
        }

    def collate_fn(self, batch):
        original_texts = [item['original'] for item in batch]
        sampled_texts = [item['sampled'] for item in batch]
        perturbed_original = [item['perturbed_original'] for item in batch]
        perturbed_sampled = [item['perturbed_sampled'] for item in batch]

        return {
            'original': original_texts,
            'sampled': sampled_texts,
            'perturbed_original': perturbed_original,
            'perturbed_sampled': perturbed_sampled
        }

    def __len__(self):
        return len(self.data)