import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from .modules import LayerNorm


class SequentialRecModel(nn.Module):
    loss_mode = "sequence_bpr"

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.item_embedding = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(args.max_seq_length, args.hidden_size)

    @property
    def candidate_embedding_weight(self):
        return self.item_embedding.weight

    @property
    def factor_embedding_weight(self):
        return self.candidate_embedding_weight

    def add_position_embedding(self, sequence):
        positions = (
            torch.arange(sequence.size(1), device=sequence.device)
            .unsqueeze(0)
            .expand_as(sequence)
        )
        hidden = self.item_embedding(sequence) + self.position_embedding(positions)
        return self.dropout(self.layer_norm(hidden))

    def get_bidirectional_mask(self, item_seq):
        mask = (item_seq > 0).unsqueeze(1).unsqueeze(2)
        return (1.0 - mask.to(dtype=next(self.parameters()).dtype)) * -10000.0

    def get_causal_mask(self, item_seq):
        valid = (item_seq > 0).long().unsqueeze(1).unsqueeze(2)
        length = item_seq.size(1)
        causal = torch.tril(
            torch.ones(
                (1, 1, length, length), device=item_seq.device, dtype=torch.long
            )
        )
        mask = valid * causal
        return (1.0 - mask.to(dtype=next(self.parameters()).dtype)) * -10000.0

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
