"""
Combinatorial (brute-force) JIT optimizer.

Searches every candidate tick range and, for each, runs a line search over
liquidity, scoring candidates by simulating the actual swap. Model-free and
exact for the simulation, but slower than the analytical optimizer.

Relies on the shared Utility object for swap-simulation-based scoring
(Utility.utility_liq / Utility.set_ticks).
"""

from uniswap_utils.position import Position
from uniswap_utils.utils import tick_from_sqrt_price, get_rounded_tick
from optimization.search import ternary_search_max


class CombinatorialOptimizer:
    """Brute-force tick-range search with a per-range liquidity line search."""

    def __init__(self, utility):
        self.utility = utility
        self.swap = utility.swap

    def optimize(self, budget, opt_func=ternary_search_max, **func_args) -> dict:
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

        for a in range(start_tick, end_tick, ts):
            for b in range(a + ts, end_tick + ts, ts):
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

        return best
