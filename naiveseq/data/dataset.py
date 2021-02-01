import os
import tqdm
import torch

from torch.utils.data import Dataset
from .dictionary import Dictionary


class Seq2SeqDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        src_data_file: str,
        tgt_data_file: str,
        src_dict: Dictionary,
        tgt_dict: Dictionary,
        max_seq_length: int=128,
        max_dataset_size: int=10000,
    ):
        self.data_dir = data_dir
        self.src_data_file = src_data_file
        self.tgt_data_file = tgt_data_file
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.max_seq_length = max_seq_length
        self.max_dataset_size = max_dataset_size
        self._read_data_file()

    def _read_data_file(self):
        self.src_sentences, self.tgt_sentences = [], []
        with open(os.path.join(self.data_dir, self.src_data_file), 'r', encoding='utf-8') as f_src, \
             open(os.path.join(self.data_dir, self.tgt_data_file), 'r', encoding='utf-8') as f_tgt:
            for i, (src_line, tgt_line) in tqdm(enumerate(zip(f_src, f_tgt)), desc=f"Reading Raw Data Files: "):
                if i >= self.max_dataset_size:
                    break
                src_ids_tensor = self.src_dict.sentence2ids(src_line.lower())
                tgt_ids_tensor = self.tgt_dict.sentence2ids(tgt_line.lower())
                src_ids_tensor = torch.cat([torch.LongTensor(
                    [self.src_dict.bos()], src_ids_tensor, torch.LongTensor([self.src_dict.eos()]))])
                tgt_ids_tensor = torch.cat([torch.LongTensor(
                    [self.tgt_dict.bos()], tgt_ids_tensor, torch.LongTensor([self.tgt_dict.eos()]))])
                self.src_sentences.append(src_ids_tensor[:self.max_seq_length])
                self.tgt_sentences.append(tgt_ids_tensor[:self.max_seq_length])
        assert len(self.src_sentences) == len(self.tgt_sentences)

    def __getitem__(self, index):
        return self.src_sentences[index], self.tgt_sentences[index]

    def __len__(self):
        return len(self.src_sentences)


class MaskPredictDataset(Seq2SeqDataset):

    def __init__(self):
        pass

    def __getitem__(self, index):
        pass
