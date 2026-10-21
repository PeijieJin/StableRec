import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def check_path(path):
    os.makedirs(path, exist_ok=True)


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _no_improvement(self, score):
        return all(score[i] <= self.best_score[i] + self.delta for i in range(len(score)))

    def __call__(self, score, model):
        if self.best_score is None or not self._no_improvement(score):
            self.best_score = score
            torch.save(model.state_dict(), self.checkpoint_path)
            self.counter = 0
            if self.verbose:
                print("Validation score increased. Saving model ...")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            self.early_stop = self.counter >= self.patience


def _rating_matrix(user_seq, num_users, num_items, leave_out):
    row, col = [], []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-leave_out]:
            row.append(user_id)
            col.append(item)
    data = np.ones(len(row), dtype=np.int8)
    return csr_matrix((data, (np.asarray(row), np.asarray(col))), shape=(num_users, num_items))


def get_user_seqs(data_file, args):
    if args.data_name in {"Yelp", "Electronics"}:
        with open(os.path.join(args.data_dir, args.data_name + "_user_seq.pkl"), "rb") as handle:
            user_seq = pickle.load(handle)
        max_item = 94304 if args.data_name == "Yelp" else 159729
    else:
        user_seq, item_set = [], set()
        with open(data_file, encoding="utf-8") as handle:
            for line in handle:
                _, raw_items = line.strip().split("\t", 1)
                items = [int(item) for item in raw_items.split("\t")]
                user_seq.append(items)
                item_set.update(items)
        max_item = max(item_set)
    num_users = len(user_seq)
    num_items = max_item + 2
    return (
        user_seq,
        max_item,
        _rating_matrix(user_seq, num_users, num_items, 2),
        _rating_matrix(user_seq, num_users, num_items, 1),
    )


def recall_at_k(actual, predicted, topk):
    total = 0.0
    for answers, predictions in zip(actual, predicted):
        total += len(set(answers) & set(predictions[:topk])) / max(1, len(set(answers)))
    return total / len(actual)


def ndcg_at_k(actual, predicted, topk):
    total = 0.0
    for answers, predictions in zip(actual, predicted):
        answer_set = set(answers)
        total += sum(int(item in answer_set) / math.log2(rank + 2) for rank, item in enumerate(predictions[:topk]))
    return total / len(actual)


def get_topk_distance_factors(item_embeddings, top_k_dist, chunk_size=1024):
    num_items = item_embeddings.size(0)
    if num_items <= 1:
        return (
            torch.empty((num_items, 0), dtype=torch.long, device=item_embeddings.device),
            torch.empty((num_items, 0), dtype=item_embeddings.dtype, device=item_embeddings.device),
        )
    k = min(top_k_dist, num_items - 1)
    normalized = F.normalize(item_embeddings, p=2, dim=1)
    all_indices, all_values = [], []
    for start in range(0, num_items, max(1, int(chunk_size))):
        end = min(start + chunk_size, num_items)
        similarities = normalized[start:end] @ normalized.transpose(0, 1)
        similarities[torch.arange(end - start, device=item_embeddings.device), torch.arange(start, end, device=item_embeddings.device)] = -float("inf")
        values, indices = torch.topk(similarities, k=k, dim=1)
        all_indices.append(indices)
        all_values.append(torch.clamp(values, min=0.0))
    return torch.cat(all_indices), torch.cat(all_values)


def get_global_popularity(user_seq, item_size):
    counts = np.zeros(item_size)
    for sequence in user_seq:
        for item in sequence:
            if item > 0:
                counts[item] += 1
    popularity = torch.tensor(counts, dtype=torch.float32)
    if popularity.max() > 0:
        popularity = torch.log1p(popularity)
        popularity = popularity / popularity.max()
    return popularity
