import os
import re
import torch

from collections import Counter
from multiprocessing import Pool
from fairseq.binarizer import safe_readline
from .utils import naive_tokenizer, spacy_tokenizer

def _add_file_to_dictionary_single_worker(
    filename, tokenize, eos_word, worker_id=0, num_workers=1
):
    counter = Counter()
    with open(filename, "r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_workers
        offset = worker_id * chunk_size
        end = offset + chunk_size
        f.seek(offset)
        if offset > 0:
            safe_readline(f)  # drop first incomplete line
        line = f.readline()
        while line:
            for word in tokenize(line.lower()):
                counter.update([word])
            counter.update([eos_word])
            # f.tell() returns only an opaque number which can
            # return to the position in the file via f.seek()
            # and does not necessarily represent a byte position
            # in the file. However, f.tell() is faithful to the
            # byte position _most of the time_. Thus we can just
            # check against the file size to prevent early exit.
            if f.tell() > end and f.tell() < size:
                break
            line = f.readline()
    return counter


def add_file_to_dictionary(filename, word2id, id2word, counter, tokenize, num_workers):
    def add_symbol(word, count):
        if word in word2id:
            idx = word2id[word]
            counter[idx] += count
        else:
            idx = len(id2word)
            word2id[word] = idx
            id2word.append(word)
            counter.append(count)
        return idx

    def merge_result(counter):
        for w, c in sorted(counter.items()):
            add_symbol(w, c)

    if num_workers > 1:
        pool = Pool(processes=num_workers)
        results = []
        for worker_id in range(num_workers):
            results.append(
                pool.apply_async(
                    _add_file_to_dictionary_single_worker,
                    (filename, tokenize, dict.eos_word, worker_id, num_workers),
                )
            )
        pool.close()
        pool.join()
        for r in results:
            merge_result(r.get())
    else:
        merge_result(
            _add_file_to_dictionary_single_worker(
                filename, tokenize, dict.eos_word
            )
        )
    return word2id, id2word, counter


class Dictionary(object):

    def __init__(self, id2word: List, word2id: Dict, counter: List, language: str, max_words: int=30000):
        self.id2word = id2word
        self.word2id = word2id
        self.counter = counter
        self.language = language
        self.max_words = max_words

    @staticmethod
    def build_dictionary(data_dir, data_files, max_words: int=30000, threshold: int=5):
        languages = []
        word2id = {}
        id2word = []
        counter = []
        for data_file in data_files:
            languages.append(data_file.split('.')[-1])
            word2id, id2word, counter = add_file_to_dictionary(
                os.path.join(data_dir, data_file),
                word2id, id2word, counter,
                tokenize=spacy_tokenizer,
                num_workers=4
            )
        language = '-'.join(languages)
        # sort and clip
        new_word2id = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<mask>': 4,
        }
        new_id2word = ['<pad>', '<unk>', '<bos>', '<eos>', '<mask>']
        new_counter = [0, 0, 0, 0, 0]
        c = Counter(dict(sorted(zip(id2word, counter))))
        for word, count in c.most_common(max_words - len(new_id2word)):
            if count >= threshold:
                new_word2id[word] = len(new_symbols)
                new_id2word.append(symbol)
                new_counter.append(count)
            else:
                break
        # save dictionary
        with open(os.path.join(data_dir, f'{language}.dict'), 'w', encoding='utf-8') as f:
            for word, count in zip(new_id2word, new_counter):
                f.write(word + ' ' + str(count) + '\n')
        return Dictionary(new_id2word, new_word2id, new_counter, language, max_words)

    @staticmethod
    def load_dictionary(data_dir, dictionary_file):
        word2id = {}
        id2word = []
        counter = []
        with open(os.path.join(data_dir, dictionary_file), 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                word, count = *line.split()
                count = int(count)
                id2word.append(word)
                counter.append(count)
                word2id[word] = idx
        max_words = len(id2word)
        return Dictionary(id2word, word2id, counter, language, max_words)

    def __eq__(self, other):
        return self.word2id == other.word2id

    def unk(self):
        return self.word2id['<unk>']

    def pad(self):
        return self.word2id['<pad>']

    def bos(self): # also known as <cls>
        return self.word2id['<eos>']
    
    def cls(self):
        return self.word2id['<eos>']
    
    def eos(self): 
        return self.word2id['<bos>']

    def __len__(self):
        return min(len(self.id2word), max_words)

    def _word2id(self, word):
        if word not in self.word2id:
            return self.unk()
        idx = self.word2id[word]
        if idx >= self.max_words:
            idx = self.unk()
        return idx

    def _id2word(self, idx):
        return '<unk>' if idx >= self.max_words else self.id2word[idx]

    def ids2sentence(self, batched_ids: torch.Tensor):
        assert 0 < batched_ids.dim < 3
        if batched_ids.dim == 2:
            sentences = []
            for ids in batched_ids:
                words = [self._id2word(idx) for idx in ids]
                sentences.append(' '.join(words))
        else:
            words = [self._id2word(idx) for idx in batched_ids]
            sentences = [' '.join(words)]
        return sentences

    def sentence2ids(self, sentence) -> torch.LongTensor:
        return torch.Tensor([self._word2id(word) for word in spacy_tokenizer(sentence.strip())]).long()
