import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SequentialRecModel


class CaserModel(SequentialRecModel):
    loss_mode = "last_bpr"

    def __init__(self, args):
        super().__init__(args)
        self.user_embedding = nn.Embedding(args.num_users, args.hidden_size)
        self.embedding_size = args.hidden_size
        self.n_h = args.nh
        self.n_v = args.nv
        self.max_seq_length = args.max_seq_length
        self.conv_v = nn.Conv2d(1, self.n_v, kernel_size=(self.max_seq_length, 1))
        self.conv_h = nn.ModuleList(
            nn.Conv2d(1, self.n_h, kernel_size=(height, self.embedding_size))
            for height in range(1, self.max_seq_length + 1)
        )
        vertical_dim = self.n_v * self.embedding_size
        horizontal_dim = self.n_h * self.max_seq_length
        self.fc1 = nn.Linear(vertical_dim + horizontal_dim, self.embedding_size)
        self.fc2 = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None):
        if user_ids is None:
            raise ValueError("Caser requires user_ids")
        item_emb = self.item_embedding(input_ids).unsqueeze(1)
        out_v = self.conv_v(item_emb).reshape(input_ids.size(0), -1)
        out_h = []
        for conv in self.conv_h:
            conv_out = F.relu(conv(item_emb).squeeze(3))
            out_h.append(F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2))
        conv_features = torch.cat([out_v, torch.cat(out_h, dim=1)], dim=1)
        sequence_features = F.relu(self.fc1(self.dropout(conv_features)))
        user_features = self.user_embedding(user_ids.view(-1))
        output = F.relu(self.fc2(torch.cat([sequence_features, user_features], dim=1)))
        return output.unsqueeze(1)
