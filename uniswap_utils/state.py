from dataclasses import dataclass

@dataclass
class State:
    price: float
    passive_dict: dict 
    tick_space: int
    fee_rate: float
    dec0: int
    dec1: int

