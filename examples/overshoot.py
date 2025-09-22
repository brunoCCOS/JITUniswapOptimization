from uniswap_utils.swap import Swap
from uniswap_utils.state import State
from optimization.utility import Utility
from optimization.search import ternary_search_max

if __name__ == '__main__':
    passive_dict = {}
    start_price = 1.0
    tick_space = 16
    fee_rate = 0.005
    dec0 = 18
    dec1 = 6

    state = State(
            start_price,
            passive_dict,
            tick_space,
            fee_rate,
            dec0,
            dec1
        )

    amount_in = 300
    zeroForOne = True

    swap = Swap(amount_in, zeroForOne, state)

    price0 = 1.0
    price1 = 1.0
    utility = Utility(swap,price0, price1)

    liq = 300000
    ternary_search_max(utility.utility_liq, 0, liq)
