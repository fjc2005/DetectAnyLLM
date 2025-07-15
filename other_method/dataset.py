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