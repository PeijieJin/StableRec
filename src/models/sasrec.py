import torch.nn as nn

from .base import SequentialRecModel
from .modules import LayerNorm, TransformerEncoder


class SASRecModel(SequentialRecModel):
    loss_mode = "sequence_bpr"

    def __init__(self, args):
        super().__init__(args)
        self.layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = TransformerEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None):
        attention_mask = self.get_causal_mask(input_ids)
        return self.item_encoder(self.add_position_embedding(input_ids), attention_mask)
