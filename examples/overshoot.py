import numpy as np
from uniswap_utils.swap import Swap, Position, State
from optimization.utility import Utility
from optimization.search import ternary_search_max
from uniswap_utils.utils import tick_from_sqrt_price, get_rounded_tick

# Set up the initial state
# Initial pool price is higher than market price
pool_price_sqrt = 1.004  # sqrt price X * Y = 100
tick_space = 20       # Standard tick spacing
dec0 = 18
dec1 = 18
initial_tick, _ = get_rounded_tick(tick_from_sqrt_price(pool_price_sqrt,dec0,dec1), tick_space)

# Market prices (price0/price1)
price0 = 1.004
price1 = 0.994
market_price = price0/price1  

# Set up the state
state = State(
        price=pool_price_sqrt,
        passive_dict={tick: 1_000_000 for tick in range(initial_tick, initial_tick + 10*tick_space, tick_space)},
        tick_space=tick_space,
        fee_rate=0.0005,
        dec0=dec0,
        dec1=dec1
        )
    

# Create the swap object
swap = Swap(zeroForOne=False, amount_in=5000, state=state)

# Create utility maximizer
maximizer = Utility(swap, price0, price1)

# Budget for liquidity
budget = 10_000

# Find the optimal position
optimal = maximizer.optimize(
    budget=budget,
    opt_func=ternary_search_max,
)


# Simulate the swap with the optimal position
optimal_position = Position(
    liq=optimal["liquidity"], 
    lower_tick=optimal["lower_tick"], 
    upper_tick=optimal["upper_tick"]
)
result = swap.simulate(optimal_position)

print("\nOptimal Position:")
print(f"Lower tick: {optimal['lower_tick']}")
print(f"Upper tick: {optimal['upper_tick']}")
print(f"Liquidity: {optimal['liquidity']}")
print(f"Utility: {optimal['utility']}")

print("\nSwap Results")
print(f"Initial tick: {initial_tick}")
print(f"Initial price: {pool_price_sqrt**2}")
print("Market price:", market_price)
print(f"Final price: {result['final_sqrt_price']**2}")
print(f"Final tick: {result['final_tick']}")
print(f"Passive Fees: {result['fees_passive_lp']}")
print(f"JIT Fees: {result['fees_jit_lp']}")
