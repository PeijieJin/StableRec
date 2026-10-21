import torch
import torch.nn as nn

from .base import SequentialRecModel
from .modules import LayerNorm, TransformerEncoder


class BERT4RecModel(SequentialRecModel):
    loss_mode = "last_bpr"

    def __init__(self, args):
        super().__init__(args)
        self.mask_token_id = args.item_size
        self.item_embedding = nn.Embedding(
            args.item_size + 1, args.hidden_size, padding_idx=0
        )
        self.layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = TransformerEncoder(args)
        self.apply(self.init_weights)

    @property
    def candidate_embedding_weight(self):
        return self.item_embedding.weight[: self.args.item_size]

    @property
    def factor_embedding_weight(self):
        return self.candidate_embedding_weight

    def reconstruct_next_item_input(self, input_ids):
        mask_column = torch.full(
            (input_ids.size(0), 1),
            self.mask_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, mask_column], dim=1)[:, 1:]

    def forward(self, input_ids, user_ids=None):
        masked_input = self.reconstruct_next_item_input(input_ids)
        attention_mask = self.get_bidirectional_mask(masked_input)
        return self.item_encoder(
            self.add_position_embedding(masked_input), attention_mask
        )
