from torch.utils.data import DataLoader

from .dataset import Seq2SeqDataset, MaskPredictDataset

def Seq2SeqDataLoader(
    dataset: Seq2SeqDataset,
    batch_size: int,
    shuffle: bool=False,
    padding_idx: int=0,
    num_workers: int=0,
    cuda: bool=True
):
    def pad_collate(sentences):
        size = max(v.size(0) for v in sentences)
        init_tensor = sentences[0].new(len(sentences), size).fill_(padding_idx)
        for i, v in enumerate(sentences):
            init_tensor[i][:len(v)] = v
        if cuda:
            init_tensor = init_tensor.cuda()
        return init_tensor
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=pad_collate,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
