"""
Combinatorial (brute-force) JIT optimizer.

Searches every candidate tick range and, for each, runs a line search over
liquidity, scoring candidates by simulating the actual swap. Model-free and
exact for the simulation, but slower than the analytical optimizer.

Relies on the shared Utility object for swap-simulation-based scoring
(Utility.utility_liq / Utility.set_ticks).
"""

import sys

from uniswap_utils.position import Position
from uniswap_utils.utils import tick_from_sqrt_price, get_rounded_tick
from optimization.search import ternary_search_max

# Cap on how many candidate positions the brute-force search evaluates per swap.
# Positions are tried smallest-first (all width-1 ranges, then width-2, ...), so
# the narrow ranges where the optimum provably sits are always covered before the
# cap bites; only the wide ranges (which never win) get trimmed. Bounds the
# O(N^2) range enumeration for swaps whose no-JIT span is hundreds of ticks.
MAX_NUMBER_POS = 10000


class CombinatorialOptimizer:
    """Brute-force tick-range search with a per-range liquidity line search."""

    def __init__(self, utility):
        self.utility = utility
        self.swap = utility.swap

    def optimize(self, budget, opt_func=ternary_search_max,
                 max_number_pos=MAX_NUMBER_POS, **func_args) -> dict:
        """Return the best position {lower_tick, upper_tick, liquidity}.

        Utility is used internally (via simulation) to rank candidates, but is
        not returned; the caller scores the chosen position separately.
        """
        u = self.utility
        state = self.swap.state
        ts = state.tick_space

        # Counterfactual final tick without JIT, and the current (start) tick.
        end_tick = self.swap.simulate(Position(0, 0, 0))["final_tick"]
        current_tick = tick_from_sqrt_price(state.price, state.dec0, state.dec1)
        start_tick, _ = get_rounded_tick(current_tick, ts)

        best = {"lower_tick": None, "upper_tick": None, "liquidity": None}
        best_utility = float("-inf")

        # Candidate ranges span from the current price to the no-JIT final price.
        # For an upward swap end_tick > start_tick; for a downward swap it is
        # below, so bound the enumeration with min/max to cover either direction
        # (a plain range(start, end) would be empty for downward swaps).
        end_r = (end_tick // ts) * ts
        lo, hi = min(start_tick, end_r), max(start_tick, end_r)

        # Enumerate by increasing position size: all width-1 ranges first, then
        # width-2, and so on. Within a width, start from the lower ticks nearest
        # the no-JIT final price (the region the swap actually reaches). This is
        # the same brute-force search over the same candidates -- only the order
        # changes -- so that when max_number_pos caps the run, the small ranges
        # (where the optimum provably lies) have already been tried and only the
        # wide ranges (which never win) are skipped.
        lowers = sorted(range(lo, hi + ts, ts), key=lambda x: abs(x - end_r))
        max_width = (hi - lo) // ts + 1

        count = 0
        truncated = False
        for width in range(1, max_width + 1):
            if truncated:
                break
            for a in lowers:
                b = a + width * ts
                if b > hi + ts:
                    continue
                if count >= max_number_pos:
                    truncated = True
                    break

                u.set_ticks(a, b)

                # Max liquidity affordable in [a, b] with the given budget.
                max_liq = Position(0, a, b).liqudity_from_budget(
                    budget, state.price, u.price0, u.price1, state.dec0, state.dec1
                )

                # Line search over liquidity, scoring via swap simulation.
                opt_utility, opt_liq = opt_func(u.utility_liq, 0, max_liq, **func_args)

                if opt_utility > best_utility:
                    best_utility = opt_utility
                    best = {"lower_tick": a, "upper_tick": b, "liquidity": opt_liq}

                count += 1

        if truncated:
            print(f"[combinatorial] max_number_pos={max_number_pos} reached at "
                  f"width {width}/{max_width}; wider positions skipped "
                  f"(swept span {(hi - lo) // ts} ticks).", file=sys.stderr)

        return best
