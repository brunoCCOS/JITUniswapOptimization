import argparse
from uniswap_utils.swap import Swap
from uniswap_utils.position import Position

def utility(
        swap: Swap, 
        position: Position,
        price0: float,
        price1: float,
        ):

    init_amt0, init_amt1 = position.tokens(swap.start_price, swap.dec0, swap.dec1)
    init_value = init_amt0 * price0 + init_amt1 * price1

    sim_max = swap.simulate(position)
    
    end_amt0, end_amt1 = position.tokens(sim_max["final_sqrt_price"], swap.dec0, swap.dec1)
    end_value = end_amt0 * price0 + end_amt1 * price1
    price_impact = end_value - init_value
    fees = (
        sim_max["fees_jit_lp"] * price0 
        if swap.zeroForOne
        else sim_max["fees_jit_lp"] * price1
    )
    utility_max = price_impact + fees
    return utility_max

def main():
    #TODO
    pass
