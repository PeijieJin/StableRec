# StableRec

A PyTorch implementation of sequential recommendation with similarity and
popularity auxiliary objectives. The code includes GRU4Rec, Caser, SASRec, and
BERT4Rec backbones with a shared StableRec wrapper and trainer.

## Directory Structure

```text
StableRec/
|-- data/
|   |-- Beauty.txt
|   |-- Home.txt
|   |-- Sports.txt
|   `-- Yelp.txt
|-- src/
|   |-- __init__.py
|   |-- datasets.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- base.py
|   |   |-- bert4rec.py
|   |   |-- caser.py
|   |   |-- gru4rec.py
|   |   |-- modules.py
|   |   |-- sasrec.py
|   |   `-- stable.py
|   |-- trainer.py
|   `-- utils.py
|-- .gitignore
|-- environment.yml
`-- README.md
```

`base.py` defines the shared sequence-model interface. `modules.py` implements
the Transformer components, `stable.py` adds the auxiliary prediction heads,
and `trainer.py` implements the recommendation loss, auxiliary losses, and ranking
evaluation. The empty dataset and utility modules are retained from the supplied
archive.

## Environment

The supplied `environment.yml` specifies:

| Dependency | Version |
| --- | --- |
| Python | 3.9.7 |
| PyTorch | 1.8.1 |
| CUDA toolkit | 11.1 |
| NumPy | 1.24.3 |
| SciPy | 1.11.1 |
| tqdm | 4.65.0 |

From the repository root, create and activate the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate envs
```

The environment specification is preserved as provided; dependency installation
and runtime compatibility have not been verified for this snapshot. The trainer
selects CUDA when available unless `args.no_cuda` is true; otherwise it uses CPU.

## Data

Each file contains one user sequence per line. Fields are tab-separated: the
first field is the user ID, followed by item IDs in their stored order.

```text
user_id<TAB>item_id_1<TAB>item_id_2<TAB>...<TAB>item_id_n
```

Statistics computed from the files included in this repository:

| Dataset | User sequences | Unique items | Interactions |
| --- | ---: | ---: | ---: |
| Beauty | 22,363 | 12,101 | 198,502 |
| Home | 66,519 | 28,237 | 551,682 |
| Sports | 35,598 | 18,357 | 296,337 |
| Yelp | 213,170 | 94,304 | 3,277,932 |

Packaged user and item IDs start at 1. The models reserve item ID 0 for padding,
so `item_size` must include that padding index. Caser also requires `num_users`
to cover the user IDs supplied to its embedding. The archive does not include
the preprocessing, data-splitting, or negative-sampling implementation.

## Run the Code

### Available Backbones

Use the following keys with `MODEL_DICT`:

| Key | Backbone | Recommendation objective |
| --- | --- | --- |
| `gru4rec` | GRU encoder | BPR over valid sequence positions |
| `caser` | Convolutional sequence encoder with user embeddings | BPR at the final output |
| `sasrec` | Causal Transformer encoder | BPR over valid sequence positions |
| `bert4rec` | Bidirectional Transformer with an appended mask token | BPR at the final output |

The BERT4Rec adapter uses the shared BPR objective rather than masked-item
cross-entropy.

### Construct a Model

After installing the dependencies, run the following in Python from the
repository root to construct a GRU4Rec-based model. These small dimensions and
weights illustrate the interface; they are not dataset-specific experiment settings.

```python
from types import SimpleNamespace

import torch

from src import MODEL_DICT, StableRecWrapper

args = SimpleNamespace(
    item_size=100,
    hidden_size=64,
    max_seq_length=5,
    dropout_prob=0.2,
    lambda_sim=0.1,
    lambda_pop=0.1,
)

backbone = MODEL_DICT["gru4rec"](args)
model = StableRecWrapper(backbone, args)
input_ids = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
sequence_output, sim_feature, pop_feature = model(input_ids)
print(sequence_output.shape, sim_feature.shape, pop_feature.shape)
```

Additional backbone settings are:

| Backbone | Required settings beyond `item_size`, `hidden_size`, and `max_seq_length` |
| --- | --- |
| GRU4Rec | `dropout_prob` |
| Caser | `num_users`, `nh`, `nv`, `hidden_dropout_prob`, `initializer_range` |
| SASRec / BERT4Rec | `hidden_dropout_prob`, `attention_probs_dropout_prob`, `num_attention_heads`, `num_hidden_layers`, `hidden_act`, `initializer_range` |

For Transformer backbones, `hidden_size` must be divisible by
`num_attention_heads`. Supported `hidden_act` values are `gelu`, `relu`, `swish`,
and `tanh`. Caser needs input sequences of length `max_seq_length` and a
`user_ids` tensor in its forward call.

### Training and Evaluation

The shared trainer computes:

```text
total_loss = BPR_loss + lambda_sim * similarity_MSE + lambda_pop * popularity_MSE
```

The similarity head predicts the mean matched distance from the supplied top-k
item-distance lookup for nonpadding input items. The popularity head predicts
the supplied target-item popularity. A head is created when its corresponding
weight is positive when the wrapper is constructed; setting both weights to zero
disables both auxiliary heads.

To run experiments, the following components must first be supplied:

1. A dataset implementation that provides batches of
   `(user_ids, input_ids, target_pos, target_neg, answers)`, including sequence
   padding, train/validation/test splits, and negative sampling.
2. `recall_at_k` and `ndcg_at_k` in `src/utils.py`, imported by `src/trainer.py`.
   Until these are implemented, importing the trainer fails.
3. A training entry point that builds the model, data loaders, argument object,
   and any required statistics, and then calls the trainer methods.

In addition to model settings, the trainer expects these attributes on `args`:

| Attributes | Purpose |
| --- | --- |
| `lr`, `weight_decay` | Adam optimizer settings |
| `no_cuda` | Device selection |
| `lambda_sim`, `lambda_pop` | Auxiliary-loss weights |
| `log_file` | Path used for appended training and evaluation logs |
| `train_matrix` | Sparse user-item matrix, indexed by user ID, for masking seen items during evaluation |
| `M_dist_indices`, `M_dist_values` | Top-k item indices and distance values, needed when the similarity loss is enabled |
| `global_pop` | Item-indexed popularity values, needed when the popularity loss is enabled |

Use integer tensors for embedding indices. The distance lookup and popularity
tensors must be compatible with the trainer's selected device. The caller must
create the log file's parent directory if needed.

After these components are available, `StableRecTrainer` exposes `train(epoch)`,
`valid(epoch)`, `test(epoch)`, and `load(file_name)`. No command-line training
command or validated experiment hyperparameters are supplied in this snapshot.

## Log Files

Training appends `epoch`, `L_BPR`, `L_sim_raw`, `L_pop_raw`, and `L_Total` to
`args.log_file`. Evaluation scores the candidate item embeddings, masks items
specified by `args.train_matrix`, and reports `HIT@5/10/20` and `NDCG@5/10/20`
using the metric functions supplied in `src/utils.py`. The `HIT` fields call
`recall_at_k` in the trainer.

No experiment logs or pretrained checkpoints are included. The trainer can load
a state dictionary with `load(file_name)`; checkpoint saving and experiment
orchestration must be handled by the caller.
