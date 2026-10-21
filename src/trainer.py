import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, *args, **kwargs):
        return iterable

from .utils import ndcg_at_k, recall_at_k


class StableRecTrainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.optim = Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        print(
            "Total Parameters:",
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def train(self, epoch):
        return self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location=self.device))

    def recommendation_loss(self, sequence_output, pos_ids, neg_ids):
        if self.model.loss_mode == "last_bpr":
            sequence_embedding = sequence_output[:, -1, :]
            pos_ids = pos_ids[:, -1]
            neg_ids = neg_ids[:, -1]
            pos_embedding = self.model.item_embedding(pos_ids)
            neg_embedding = self.model.item_embedding(neg_ids)
            valid = (pos_ids > 0).float()
        else:
            sequence_embedding = sequence_output.reshape(-1, self.args.hidden_size)
            pos_embedding = self.model.item_embedding(pos_ids).reshape(
                -1, self.args.hidden_size
            )
            neg_embedding = self.model.item_embedding(neg_ids).reshape(
                -1, self.args.hidden_size
            )
            valid = (pos_ids > 0).reshape(-1).float()

        pos_logits = torch.sum(pos_embedding * sequence_embedding, dim=-1)
        neg_logits = torch.sum(neg_embedding * sequence_embedding, dim=-1)
        return torch.sum(
            -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24) * valid
        ) / valid.sum().clamp_min(1.0)

    def predict_full(self, sequence_output):
        return sequence_output @ self.model.candidate_embedding_weight.transpose(0, 1)

    def lookup_topk_dist(self, input_ids, target_items):
        knn_indices = self.args.M_dist_indices[input_ids]
        knn_values = self.args.M_dist_values[input_ids]
        if knn_indices.size(-1) == 0:
            return torch.zeros_like(input_ids, dtype=knn_values.dtype)
        matches = (knn_indices == target_items.view(-1, 1, 1)).to(knn_values.dtype)
        return torch.max(knn_values * matches, dim=-1).values

    def _train_epoch(self, epoch, dataloader):
        self.model.train()
        totals = {"rec": 0.0, "sim": 0.0, "pop": 0.0, "all": 0.0}
        for _, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            user_ids, input_ids, target_pos, target_neg, _ = (
                tensor.to(self.device) for tensor in batch
            )
            sequence_output, sim_feature, pop_feature = self.model(
                input_ids, user_ids=user_ids
            )
            rec_loss = self.recommendation_loss(sequence_output, target_pos, target_neg)
            target_items = target_pos[:, -1]
            sim_loss = torch.tensor(0.0, device=self.device)
            pop_loss = torch.tensor(0.0, device=self.device)

            if self.args.lambda_sim > 0.0 and sim_feature is not None:
                sim_score = sim_feature[:, -1, :].squeeze(-1)
                item_dist = self.lookup_topk_dist(input_ids, target_items)
                mask = (input_ids > 0).float()
                valid_length = mask.sum(dim=-1) + 1e-8
                dist_penalty = torch.sum(item_dist * mask, dim=-1) / valid_length

                sim_loss = F.mse_loss(sim_score, dist_penalty.detach())

            if self.args.lambda_pop > 0.0 and pop_feature is not None:
                pop_score = pop_feature[:, -1, :].squeeze(-1)
                target_pop = self.args.global_pop[target_items]

                pop_loss = F.mse_loss(pop_score, target_pop.detach())

            total_loss = (
                rec_loss
                + self.args.lambda_sim * sim_loss
                + self.args.lambda_pop * pop_loss
            )
            self.optim.zero_grad()
            total_loss.backward()
            self.optim.step()
            totals["rec"] += rec_loss.item()
            totals["sim"] += sim_loss.item()
            totals["pop"] += pop_loss.item()
            totals["all"] += total_loss.item()

        count = max(1, len(dataloader))
        result = {
            "epoch": epoch,
            "L_BPR": f"{totals['rec'] / count:.4f}",
            "L_sim_raw": f"{totals['sim'] / count:.4f}",
            "L_pop_raw": f"{totals['pop'] / count:.4f}",
            "L_Total": f"{totals['all'] / count:.4f}",
        }
        print(result)
        with open(self.args.log_file, "a", encoding="utf-8") as handle:
            handle.write(str(result) + "\n")
        return totals["all"] / count

    def _evaluate(self, epoch, dataloader):
        self.model.eval()
        predictions, answers = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                user_ids, input_ids, _, _, batch_answers = (
                    tensor.to(self.device) for tensor in batch
                )
                sequence_output, _, _ = self.model(input_ids, user_ids=user_ids)
                scores = (
                    self.predict_full(sequence_output[:, -1, :]).cpu().numpy().copy()
                )
                seen = self.args.train_matrix[user_ids.cpu().numpy()].toarray() > 0
                scores[seen] = -np.inf
                topk = min(20, scores.shape[1])
                indices = np.argpartition(scores, -topk, axis=1)[:, -topk:]
                values = scores[np.arange(len(scores))[:, None], indices]
                order = np.argsort(values, axis=1)[:, ::-1]
                predictions.append(indices[np.arange(len(scores))[:, None], order])
                answers.append(batch_answers.cpu().numpy())

        prediction_array = np.concatenate(predictions)
        answer_array = np.concatenate(answers)
        result = {"Epoch": epoch}
        score_list = []
        for k in (5, 10, 20):
            hit = recall_at_k(answer_array, prediction_array, k)
            ndcg = ndcg_at_k(answer_array, prediction_array, k)
            result[f"HIT@{k}"] = f"{hit}"
            result[f"NDCG@{k}"] = f"{ndcg}"
            score_list.extend([hit, ndcg])
        print(result)
        with open(self.args.log_file, "a", encoding="utf-8") as handle:
            handle.write(str(result) + "\n")
        return score_list, str(result)

    def iteration(self, epoch, dataloader, train=True):
        if train:
            return self._train_epoch(epoch, dataloader)
        return self._evaluate(epoch, dataloader)
