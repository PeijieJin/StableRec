import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_


class GRU4RecModel(nn.Module):
    loss_mode = "sequence_bpr"

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_seq_length = args.max_seq_length
        self.item_embedding = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(args.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=128,
            num_layers=1,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(128, args.hidden_size)
        self.apply(self._init_weights)

    @property
    def candidate_embedding_weight(self):
        return self.item_embedding.weight

    @property
    def factor_embedding_weight(self):
        return self.item_embedding.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, input_ids, user_ids=None):
        hidden, _ = self.gru_layers(self.emb_dropout(self.item_embedding(input_ids)))
        return self.dense(hidden)
