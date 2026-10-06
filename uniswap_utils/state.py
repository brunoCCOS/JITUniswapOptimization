from dataclasses import dataclass
from uniswap_utils import Numerical

@dataclass
class State:
    price_sqrt: Numerical
    passive_dict: dict[int, Numerical] 
    tick_space: int
    fee_rate: Numerical
    dec0: int
    dec1: int
    tick_idx_offset: int = 0

