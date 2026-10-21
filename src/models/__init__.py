from .bert4rec import BERT4RecModel
from .caser import CaserModel
from .gru4rec import GRU4RecModel
from .sasrec import SASRecModel
from .stable import StableRecWrapper


MODEL_DICT = {
    "gru4rec": GRU4RecModel,
    "caser": CaserModel,
    "sasrec": SASRecModel,
    "bert4rec": BERT4RecModel,
}

__all__ = ["MODEL_DICT", "StableRecWrapper"]
