"""Datasets for sequence-output and final-output backbones."""

import torch
from torch.utils.data import Dataset

from .utils import neg_sample


def _pad(values, max_len):
    return ([0] * max(0, max_len - len(values)) + values)[-max_len:]


class SequenceBPRDataset(Dataset):
    """The original StableRec next-item dataset used by GRU4Rec and SASRec."""

    def __init__(self, args, user_seq, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        items = self.user_seq[index]
        if self.data_type == "train":
            input_ids, target_pos, answer = items[:-3], items[1:-2], [0]
        elif self.data_type == "valid":
            input_ids, target_pos, answer = items[:-2], items[1:-1], [items[-2]]
        else:
            input_ids, target_pos, answer = items[:-1], items[1:], [items[-1]]

        item_set = set(items)
        target_neg = [neg_sample(item_set, self.args.item_size) for _ in input_ids]
        return (
            torch.tensor(index, dtype=torch.long),
            torch.tensor(_pad(input_ids, self.max_len), dtype=torch.long),
            torch.tensor(_pad(target_pos, self.max_len), dtype=torch.long),
            torch.tensor(_pad(target_neg, self.max_len), dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )


class FinalBPRDataset(Dataset):
    """Sliding-prefix training for Caser and terminal-mask BERT4Rec."""

    def __init__(self, args, user_seq, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.data_type = data_type
        self.max_len = args.max_seq_length
        self.examples = []
        if data_type == "train":
            for user_id, items in enumerate(user_seq):
                train_sequence = items[:-2]
                for end in range(2, len(train_sequence) + 1):
                    self.examples.append((user_id, train_sequence[: end - 1], train_sequence[end - 1], items))
        else:
            for user_id, items in enumerate(user_seq):
                if data_type == "valid":
                    self.examples.append((user_id, items[:-2], items[-2], items))
                else:
                    self.examples.append((user_id, items[:-1], items[-1], items))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        user_id, history, positive, full_sequence = self.examples[index]
        negative = neg_sample(set(full_sequence), self.args.item_size)
        target_pos = [0] * (self.max_len - 1) + [positive]
        target_neg = [0] * (self.max_len - 1) + [negative]
        answer = [0] if self.data_type == "train" else [positive]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(_pad(history, self.max_len), dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(target_neg, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )


def build_dataset(args, user_seq, data_type):
    if args.model_name.lower() in {"caser", "bert4rec"}:
        return FinalBPRDataset(args, user_seq, data_type)
    return SequenceBPRDataset(args, user_seq, data_type)
