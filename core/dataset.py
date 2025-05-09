from torch.utils.data import Dataset
import json


class CustomDataset(Dataset):
    def __init__(self, data_json_dir, MIRAGE: bool = False):
        self.data_dir = data_json_dir
        self.data = json.load(open(data_json_dir, 'r', encoding='utf-8'))
        self.MIRAGE_format = MIRAGE

    def __len__(self):
        return len(self.data) if self.MIRAGE_format else len(self.data['original'])

    def __getitem__(self, index):
        if self.MIRAGE_format:
            original_text = self.data[index]['original']
            sampled_text = self.data[index]['rewritten']
        else:
            original_text = self.data['original'][index]
            sampled_text = self.data['rewritten'][index]

        return {'original': original_text, 'rewritten': sampled_text}
