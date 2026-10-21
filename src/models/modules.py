import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        return (
            self.weight * (x - mean)
            / torch.sqrt(variance + self.variance_epsilon)
            + self.bias
        )


class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense_1 = nn.Linear(args.hidden_size, 4 * args.hidden_size)
        self.dense_2 = nn.Linear(4 * args.hidden_size, args.hidden_size)
        self.layer_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.hidden_act = args.hidden_act

    def _activate(self, x):
        if self.hidden_act == "gelu":
            return F.gelu(x)
        if self.hidden_act == "relu":
            return F.relu(x)
        if self.hidden_act == "swish":
            return x * torch.sigmoid(x)
        if self.hidden_act == "tanh":
            return torch.tanh(x)
        raise ValueError(f"Unsupported hidden_act: {self.hidden_act}")

    def forward(self, input_tensor):
        hidden_states = self.dense_2(self._activate(self.dense_1(input_tensor)))
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + input_tensor)


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_heads = args.num_attention_heads
        self.head_size = args.hidden_size // args.num_attention_heads
        self.all_head_size = self.num_heads * self.head_size
        self.scale = math.sqrt(self.head_size)
        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def _reshape(self, x):
        return x.view(*x.size()[:-1], self.num_heads, self.head_size)

    def forward(self, input_tensor, attention_mask):
        query = self._reshape(self.query(input_tensor)).permute(0, 2, 1, 3)
        key = self._reshape(self.key(input_tensor)).permute(0, 2, 3, 1)
        value = self._reshape(self.value(input_tensor)).permute(0, 2, 1, 3)
        scores = torch.matmul(query, key) / self.scale
        probs = self.attn_dropout(torch.softmax(scores + attention_mask, dim=-1))
        context = torch.matmul(probs, value).permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:-2], self.all_head_size)
        hidden_states = self.out_dropout(self.dense(context))
        return self.layer_norm(hidden_states + input_tensor)


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)

    def forward(self, hidden_states, attention_mask):
        return self.feed_forward(self.attention(hidden_states, attention_mask))


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        block = TransformerBlock(args)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(args.num_hidden_layers)]
        )

    def forward(self, hidden_states, attention_mask):
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        return hidden_states
