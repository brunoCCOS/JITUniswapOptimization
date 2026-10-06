from uniswap_utils.swap import Swap
from uniswap_utils.position import Position
from optimization.search import ternary_search_max

class Utility:
    def __init__(
            self,
            swap: Swap, 
            price0: float,
            price1: float,
            ):
        self.swap = swap
        self.price0 = price0
        self.price1 = price1

    def set_ticks(self, lower: int, upper: int):
        self.lower_tick = lower
        self.upper_tick = upper

    def set_liq(self, liq: float):
        self.liq = liq

    def utility_liq(self, liq):
        position = Position(liq, self.lower_tick, self.upper_tick)
        return self.positions_utility(position)

    def utility_tick(self, lower,upper):
        position = Position(self.liq, lower, upper)
        return self.positions_utility(position)

    def positions_utility(self, position: Position | list[Position]):
        positions = [position] if type(position) is Position else position

        init_amt0, init_amt1 = 0, 0
        for position in positions:
            pos_amt0, pos_amt1 = position.tokens(self.swap.state.price_sqrt, self.swap.state.dec0, self.swap.state.dec1)
            init_amt0 += pos_amt0
            init_amt1 += pos_amt1
        init_value = init_amt0 * self.price0 + init_amt1 * self.price1

        sim_max = self.swap.simulate(positions)

        end_amt0, end_amt1 = 0, 0
        for position in positions:
            pos_amt0, pos_amt1 = position.tokens(sim_max["final_sqrt_price"], self.swap.state.dec0, self.swap.state.dec1)
            end_amt0 += pos_amt0
            end_amt1 += pos_amt1
        end_value = end_amt0 * self.price0 + end_amt1 * self.price1

        price_impact = end_value - init_value

        fees = (
            sim_max["fees_jit_lp"] * self.price0 
            if self.swap.zeroForOne
            else sim_max["fees_jit_lp"] * self.price1
        )

        utility_max = price_impact + fees
        
        return utility_max

    def optimize(self,
                 budget,
                 method="combinatorial",
                 opt_func=ternary_search_max,
                 **func_args
                 ):
        """
        Find optimal position parameters (ticks and liquidity) within a budget.

        Dispatches to one of two interchangeable strategies to pick a position
        (lower_tick, upper_tick, liquidity):
        - "combinatorial": brute-force every tick range + line search on liquidity
          (slow, model-free).
        - "analytical": closed-form solution (Lemmas 5.1/5.2, ported from MATLAB),
          computing the optimal liquidity per candidate tick directly (fast).

        The optimizers only choose the position. The reported utility is always
        computed the same way afterward, via swap simulation (positionpositions_utility),
        so results are comparable regardless of which optimizer was used.

        Args:
            budget: Maximum amount to use for liquidity provision.
            method: "combinatorial" (default) or "analytical".
            opt_func: Search function from search.py (combinatorial method only).
            **func_args: Extra args forwarded to opt_func (combinatorial method only).

        Returns:
            Dictionary with optimal parameters (lower_tick, upper_tick, liquidity, utility).
        """
        if method in ("combinatorial", "combinatory", "numerical"):
            position = self._optimize_combinatorial(budget, opt_func, **func_args)
            positions = [Position(
                liq = position["liquidity"],
                lower_tick = position["lower_tick"],
                upper_tick = position["upper_tick"],
            )]
        elif method == "analytical":
            positions = [Position(
                liq = position["liquidity"],
                lower_tick = position["lower_tick"],
                upper_tick = position["upper_tick"],
            ) for position in self._optimize_analytical(budget)]
        else:
            raise ValueError(
                f"Unknown optimization method {method!r}; "
                "expected 'combinatorial' or 'analytical'."
            )

        return positions

    def _optimize_combinatorial(self, budget, opt_func, **func_args):
        """
        Brute-force optimizer (tick-range search + liquidity line search).

        Delegates to optimization.combinatorial.CombinatorialOptimizer, which
        scores candidates via full swap simulation using this Utility's scoring.
        """
        from optimization.combinatorial import CombinatorialOptimizer

        return CombinatorialOptimizer(self).optimize(budget, opt_func, **func_args)

    def _optimize_analytical(self, budget):
        """
        Closed-form optimizer (Lemmas 5.1 / 5.2), ported from the MATLAB code.

        Delegates to optimization.analytical.AnalyticalOptimizer, which computes
        the optimal liquidity per candidate tick range directly (no simulation).
        """
        from optimization.analytical import AnalyticalOptimizer

        return AnalyticalOptimizer(self.swap, self.price0, self.price1).optimize(budget)
