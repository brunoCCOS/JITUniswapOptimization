import sys
sys.path.append("..")

import math
import numpy as np
from decimal import Decimal
from uniswap_utils.swap import Swap, Position, State
from optimization.analytical import AnalyticalOptimizer
from uniswap_utils.utils import tick_from_sqrt_price, get_rounded_tick, sqrt_price_from_tick
from pprint import pprint
# Tests start here
tests = {
    "liquidity_to_cross_t_mbar": dict(
        test_type = "tick_crossing", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, amount_in = 900, 
            fee_rate = 0.0005, budget_dollars = 10000, zeroForOne = True, max_num_ticks = 10, 
            passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        cases = [
            dict(positions = [Position(7520, 9, 10)], expected_range = (8, 9)),
            dict(positions = [Position(7521, 9, 10)], expected_range = (9, 10)),
        ]
    ),

    "liquidity_to_cross_t_mbar-1": dict(
        test_type = "tick_crossing", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, amount_in = 2000, 
            fee_rate = 0.0005, budget_dollars = 10000, zeroForOne = True, max_num_ticks = 10, 
            passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        cases = [
            dict(positions = [Position(116220, 8,9)], expected_range = (7,8)),
            dict(positions = [Position(116221, 8,9)], expected_range = (8,9)),
        ]
    ),

    # k = 0, R > F, P >= FC/(R - F)
    "optimal_allocation-k=0-single_tick-prop3.1(a)": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, 
            amount_in = 800, fee_rate = 0.0005, budget_dollars = 10000, zeroForOne = True, 
            max_num_ticks = 10, passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        expected_allocations = [{"lower_tick": 9, "upper_tick":10, "liq": 0}]
    ),

    # k = 0, F > R, P >= FC/(R - F)
    "optimal_allocation-k=0-single_tick-prop3.1(b)": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, 
            amount_in = 100, fee_rate = 0.01, budget_dollars = 10000, zeroForOne = True, 
            max_num_ticks = 10, passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        expected_allocations = [{"lower_tick": 9, "upper_tick":10, "liq": 9965743.844035}]
    ),

    # k = 0, F > R, P < FC/(R - F), and L_inner < L_max_budget_dollars
    "optimal_allocation-k=0-single_tick-prop3.1(c)-inner": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, 
            amount_in = 800, fee_rate = 0.01, budget_dollars = 10000, zeroForOne = True, 
            max_num_ticks = 10, passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        expected_allocations = [{"lower_tick": 9, "upper_tick":10, "liq": 242923.80856919}]
    ),

    # k = 0, F > R, P < FC/(R - F), and L_inner > L_max_budget_dollars
    "optimal_allocation-k=0-single_tick-prop3.1(c)-all_budget": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, 
            amount_in = 800, fee_rate = 0.01, budget_dollars = 100, zeroForOne = True, 
            max_num_ticks = 10, passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        expected_allocations = [{"lower_tick": 9, "upper_tick":10, "liq": 99657.43844035}]
    ),

    # k = 1, R0 > F, P0 >= F * C0 / (R0 - F), Lemma 5.2 unfeasible
    "optimal_allocation-k=1-two_last_ticks-lemma5.1(a)": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, 
            amount_in = 1000, fee_rate = 0.0005, budget_dollars = 100, zeroForOne = True, 
            max_num_ticks = 10, passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        expected_allocations = [{"lower_tick": 8, "upper_tick": 9, "liq": 0}, {"lower_tick": 9, "upper_tick":10, "liq": 0}]
    ),

    # k = 1, R0 > F, P0 < F * C0 / (R0 - F), Lemma 5.2 unfeasible
    "optimal_allocation-k=1-two_last_ticks-lemma5.1(b)": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, 
            amount_in = 1800, fee_rate = 0.0075, budget_dollars = 100, zeroForOne = True, 
            max_num_ticks = 10, passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        expected_allocations = [{"lower_tick": 8, "upper_tick": 9, "liq": 0}, {"lower_tick": 9, "upper_tick":10, "liq": 0}]
    ),

    # k = 1, R0 = F, all_budget as solution, Lemma 5.2 unfeasible
    "optimal_allocation-k=1-two_last_ticks-lemma5.1(c)-all_budget": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, 
            amount_in = 1800, fee_rate = 0.0080210037290687, budget_dollars = 100, zeroForOne = True, 
            max_num_ticks = 10, passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        expected_allocations = [{"lower_tick": 8, "upper_tick": 9, "liq": 99757.140736614}, {"lower_tick": 9, "upper_tick":10, "liq": 0}]
    ),

    # k = 1, R0 = F, sqrt(P0 (C0 + P0)) as solution, Lemma 5.2 unfeasible
    "optimal_allocation-k=1-two_last_ticks-lemma5.1(c)-p0(c0+p0)": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, 
            amount_in = 1800, fee_rate = 0.0080210037290687, budget_dollars = 10000, zeroForOne = True, 
            max_num_ticks = 10, passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        expected_allocations = [{"lower_tick": 8, "upper_tick": 9, "liq": 1000457.3529893495}, {"lower_tick": 9, "upper_tick":10, "liq": 0}]
    ),

    # k = 7, 
    "optimal_allocation-k=3-lemma5.1(d)": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 1.02, dec0 = 18, dec1 = 18, tick_space = 20, price0 = 1.004, price1 = 0.994, 
            amount_in = 3000, fee_rate = 0.0005, budget_dollars = 10000, zeroForOne = True, 
            max_num_ticks = 10, passive_liquidity_per_range = 1e6, tick_idx_offset = 0
        ),
        expected_allocations = [{"lower_tick": 8, "upper_tick": 9, "liq": 1000457.3529893495}, {"lower_tick": 9, "upper_tick":10, "liq": 0}]
    ),

    "example-professor": dict(
        test_type = "optimal_allocation", 
        base_params = dict(
            pool_price = 3000.7, dec0 = 18, dec1 = 18, tick_space = 10, price0 = 3000.7, price1 = 1, 
            amount_in = 250, fee_rate = 0.0005, budget_dollars = 2*1e6, zeroForOne = True, 
            max_num_ticks = 400, passive_liquidity_per_range = 2*1e6, tick_idx_offset = 76070
        ),
        expected_allocations = [{"lower_tick": 8, "upper_tick": 9, "liq": -1}, {"lower_tick": 9, "upper_tick":10, "liq": -1}]
    ),
}
selected_test = "example-professor"
args = tests[selected_test]
test_type = args["test_type"]

pool_price = args["base_params"]["pool_price"]
dec0 = args["base_params"]["dec0"]
dec1 = args["base_params"]["dec1"]
tick_space = args["base_params"]["tick_space"]
price0 = args["base_params"]["price0"]
price1 = args["base_params"]["price1"]
amount_in = args["base_params"]["amount_in"]
fee_rate = args["base_params"]["fee_rate"]
budget_dollars = args["base_params"]["budget_dollars"]
zeroForOne = args["base_params"]["zeroForOne"]
max_num_ticks = args["base_params"]["max_num_ticks"]
tick_idx_offset = args["base_params"]["tick_idx_offset"]
passive_liquidity_per_range = args["base_params"]["passive_liquidity_per_range"]

test_cases = args.get("cases", None)
start_tick_idx = AnalyticalOptimizer._get_tick_idx_from_tick_price(pool_price, tick_idx_offset, tick_space, dec0, dec1)
step = -1

passive_dict = {tick: passive_liquidity_per_range for tick in range(start_tick_idx, start_tick_idx - max_num_ticks, step)}

state = State(
    price_sqrt = math.sqrt(pool_price),
    passive_dict = passive_dict, 
    tick_idx_offset = tick_idx_offset,
    tick_space = tick_space, 
    fee_rate = fee_rate, 
    dec0 = dec0,
    dec1 = dec1
)

F = 1 + state.fee_rate

swap = Swap(amount_in, zeroForOne, state)
maximizer = AnalyticalOptimizer(swap, price0, price1)
end_price, end_range = maximizer.simulate_swap(
    Decimal(pool_price), passive_dict, {}, amount_in, tick_idx_offset, tick_space, dec0, dec1
)
end_tick_idx_no_jit = end_range[0]

ranges = []
hi = start_tick_idx + tick_space
while hi > end_tick_idx_no_jit:
    ranges.append((hi - tick_space, hi))
    hi -= tick_space
print(ranges)
tick_params = maximizer._precompute(ranges, amount_in, budget_dollars, price0, price1, F, pool_price, dec0, dec1)
pprint(tick_params)
# Tick Crossing Test
if test_type == "tick_crossing":
    for i, test_dict in enumerate(test_cases, start = 1):
        positions = {position.lower_tick: position.liq for position in test_dict["positions"]}
        expected_range = test_dict["expected_range"]
        final_price, final_range = maximizer.simulate_swap(
            pool_price, passive_dict, positions, amount_in, tick_idx_offset, tick_space, dec0, dec1
        )

        status = "(PASS)" if final_range == expected_range else ("(FAIL)")
        print(f"{status} -> [TEST #{i}]: Crossing ticks after JIT's allocation:", positions)
        print("Resulting range:", final_range)
        print("Expected range:", expected_range)
        print()
# Optimal Allocation
def compare_positions(actual, expected):
    if len(actual) != len(expected):
        return False

    return all(
        math.isclose(a["liq"], e["liq"])
        and a["lower_tick"] == e["lower_tick"]
        and a["upper_tick"] == e["upper_tick"]
        for a, e in zip(actual, expected)
    )

if test_type == "optimal_allocation":
    optimals = maximizer.optimize(
        budget = budget_dollars
    )
    optimal_positions = {
        position["lower_tick"]: position["liq"] for position in optimals["positions"]
    } if optimals else None
    utility = optimals["utility"]

    final_price, final_range = maximizer.simulate_swap(
        pool_price, passive_dict, optimal_positions, amount_in, tick_idx_offset, tick_space, dec0, dec1
    )

    final_tick_idx = final_range[0]
    num_crossed_ticks = start_tick_idx - final_tick_idx 
    num_crossed_ticks_no_jit = start_tick_idx - end_tick_idx_no_jit 
    
    expected_allocations = args["expected_allocations"]
    status = "(PASS)" if compare_positions(optimals["positions"], expected_allocations) else ("(FAIL)")
    print(f"{status} -> [TEST]: Optimal JIT's allocation")
    print("Resulting positions:", optimals["positions"])
    print("Expected positions:", expected_allocations)
    
    print("\tUtility:", utility)
    print(f"\tFinal price with JIT's allocation: {final_price:.4f}")
    print(f"\tFinal tick range with JIT's allocation: {final_range}")
    print(f"\tNumber of crossed ticks (No JIT): {num_crossed_ticks_no_jit}")
    print(f"\tNumber of crossed ticks (With JIT): {num_crossed_ticks}")
    print()