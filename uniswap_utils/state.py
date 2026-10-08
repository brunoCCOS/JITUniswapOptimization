"""
Immutable snapshot of a pool's state as seen by the optimizer: current
sqrt-price, the passive (non-JIT) liquidity map keyed by tick, tick spacing,
fee rate, and token decimals. Consumed by `Swap`/`Position` to simulate how a
candidate JIT position would interact with the existing passive liquidity.
"""

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