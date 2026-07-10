"""
Analytical (closed-form) JIT optimizer, ported from the MATLAB implementation.

For each candidate tick range the optimal JIT liquidity is computed directly via
Lemmas 5.1 / 5.2 (no swap simulation, no line search), then the best range is
selected. j=0 is the range containing the initial price; j increases toward the
counterfactual final price (the price the swap would reach without JIT).

The MATLAB derivation is for a downward (zeroForOne) swap: token0 in, price
falling, JIT liquidity placed below the price and funded with token1. Rather
than mirror every formula, an upward (oneForZero) swap is reframed once into
those same "canonical" coordinates by swapping the token roles and inverting the
prices (sqrt(P) -> 1/sqrt(P)). The body then runs a single direction-agnostic
code path. Only the reported tick range is mapped back to the pool's frame.

Note: this optimizes the paper's closed-form utility model, which is a different
objective from the simulation-based utility used by the combinatorial optimizer
(see optimization.utility.Utility._optimize_combinatorial). The two agree on the
model's own terms but need not produce identical numbers.
"""

import math
from dataclasses import dataclass

from uniswap_utils.swap import Swap
from uniswap_utils.position import Position
from uniswap_utils.utils import (
    tick_from_sqrt_price,
    get_rounded_tick,
    sqrt_price_from_tick,
)


@dataclass
class TickParams:
    """Per-range parameters feeding the closed-form solution."""

    lower: int
    upper: int
    P: float          # passive liquidity active in the range
    dx: float         # remaining trade when the swap reaches this range
    R: float          # entry price x (py/px)
    C: float          # capacity parameter, dx x sqrt(entry price)
    A: float          # allocation factor
    L_inner: float    # interior (first-order) solution
    L0: float         # minimum liquidity to absorb the remaining trade
    L_max: float      # budget cap
    cap_per_L: float  # input token per L across the full range
    traversed_cap: float  # input token per L actually traversed (partial for j=0)
    fc_slope: float   # fully-crossed (linear) utility per unit L over the
                      # traversed slice; used when L < L0 (swap not contained)


class AnalyticalOptimizer:
    """Closed-form JIT liquidity optimizer (Lemmas 5.1 / 5.2)."""

    def __init__(self, swap: Swap, price0: float, price1: float):
        self.swap = swap
        self.price0 = price0
        self.price1 = price1

    # ------------------------------------------------------------------ #

    def optimize(self, budget) -> dict:
        """Return {lower_tick, upper_tick, liquidity, utility} for the best range."""
        state = self.swap.state
        ts = state.tick_space
        dec0, dec1 = state.dec0, state.dec1
        F = 1.0 + float(state.fee_rate)
        Delta_x = float(self.swap.amount_in)
        direction_up = not self.swap.zeroForOne

        # Reframe into canonical (downward) coordinates. px/py are the USD prices
        # of the input/output tokens; canon_sqrt maps a tick to its sqrt price in
        # the canonical frame (identity for a down swap, inverted for an up swap).
        pool_sqrt = float(state.price)
        if direction_up:
            px, py = float(self.price1), float(self.price0)  # in=token1, out=token0
            init_sqrt = 1.0 / pool_sqrt
            canon_sqrt = lambda t: 1.0 / float(sqrt_price_from_tick(t, dec0, dec1))
        else:
            px, py = float(self.price0), float(self.price1)  # in=token0, out=token1
            init_sqrt = pool_sqrt
            canon_sqrt = lambda t: float(sqrt_price_from_tick(t, dec0, dec1))

        # Budget in units of the token the JIT LP deposits (the output token py),
        # mirroring MATLAB's B, where L_max = B / eps.
        B_tokens = budget / py

        current_tick = tick_from_sqrt_price(state.price, dec0, dec1)
        start_tick, _ = get_rounded_tick(current_tick, ts)
        end_tick = self.swap.simulate(Position(0, 0, 0))["final_tick"]

        ranges = self._build_ranges(start_tick, end_tick, ts, direction_up)
        if not ranges:
            # The swap stays within the current tick-space range (it does not
            # cross a range boundary). Lemma 5.1 still applies to that single
            # range and can prescribe positive liquidity, so fall back to it
            # instead of returning "no JIT". (Gating on integer-tick equality
            # wrongly skipped this whole class of swaps and made the result
            # depend on an integer-tick crossing irrelevant to the tick-space
            # ranges actually optimized over.)
            ranges = [(start_tick, start_tick + ts)]

        params = self._precompute(
            ranges, Delta_x, B_tokens, px, py, F, init_sqrt, canon_sqrt
        )
        return self._solve(params, F, px)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_ranges(start_tick, end_tick, ts, direction_up):
        """Ranges from the initial price (j=0) toward the final price."""
        ranges = []
        if direction_up:
            lo = start_tick
            while lo < end_tick:
                ranges.append((lo, lo + ts))
                lo += ts
        else:
            hi = start_tick + ts
            while hi > end_tick:
                ranges.append((hi - ts, hi))
                hi -= ts
        return ranges

    def _precompute(self, ranges, Delta_x, B_tokens, px, py, F,
                    init_sqrt, canon_sqrt) -> list[TickParams]:
        """Per-range parameters in canonical (downward) coordinates.

        In this frame the input token flows as 1/sqrt(P) and the deposited token
        as sqrt(P), so the formulas are the single MATLAB down-direction form
        regardless of the actual swap direction.
        """
        # Fee is charged on top, so the net trade that actually moves the price
        # (and that passive liquidity absorbs) is the gross amount divided by F.
        net_total = Delta_x / F

        out: list[TickParams] = []
        for j, (lower, upper) in enumerate(ranges):
            # Canonical sqrt-price bounds (sqrt_lo < sqrt_hi always).
            a, b = canon_sqrt(lower), canon_sqrt(upper)
            sqrt_lo, sqrt_hi = min(a, b), max(a, b)

            cap_per_L = 1.0 / sqrt_lo - 1.0 / sqrt_hi   # input token per L, full range
            eps_budget = sqrt_hi - sqrt_lo              # deposited token per L

            # Capacity the trade actually traverses in this range. The swap starts
            # inside range j=0, so it only crosses from the initial price to the
            # boundary, not the full range; deeper ranges are fully traversed.
            # traversed_cap is the input token (X) per L over the traversed slice;
            # traversed_eps is the deposited token (Y) per L over that same slice.
            if j == 0:
                traversed_cap = max(0.0, 1.0 / sqrt_lo - 1.0 / init_sqrt)
                traversed_eps = max(0.0, init_sqrt - sqrt_lo)
            else:
                traversed_cap = cap_per_L
                traversed_eps = eps_budget

            # Remaining trade after passive liquidity absorbs the earlier ranges,
            # using each range's actually-traversed capacity.
            dx = net_total
            for i in range(j):
                dx -= out[i].P * out[i].traversed_cap
            dx = max(0.0, dx)

            # Entry price: actual initial price for j=0, else the boundary the
            # trade first reaches (the higher sqrt price in canonical coords).
            sqrt_j = init_sqrt if j == 0 else sqrt_hi
            price_j = sqrt_j * sqrt_j

            P = float(self.swap.state.passive_dict.get(lower, 0.0))
            R = price_j * (py / px)
            C = dx * sqrt_j

            A = math.sqrt((F / R) * (P / (C + P))) if (R > 0 and C + P > 0) else 0.0
            L_inner = (C * A) / (1.0 - A) - P if A < 1.0 else float("inf")
            # For j=0 the range straddles the current price; the JIT LP must deposit
            # both tokens. Cost = token0_portion*(px/py) + token1_portion.
            # For j>0 the range is fully below current price → only token1 needed.
            if j == 0:
                token0_portion = max(0.0, 1.0 / init_sqrt - 1.0 / sqrt_hi)
                eps_actual = token0_portion * (px / py) + traversed_eps
            else:
                eps_actual = eps_budget
            L_max = B_tokens / eps_actual if eps_actual > 0 else 0.0

            # Fully-crossed (linear) utility per unit L, over the slice the swap
            # actually traverses: u_fc(L) = px*F*T - py*y with T = L*traversed_cap
            # (token X executed) and y = L*traversed_eps (token Y deposited/paid).
            # This is the paper's crossed-tick utility and is the correct value
            # when the JIT cannot contain the swap in this range (L < L0).
            fc_slope = px * F * traversed_cap - py * traversed_eps

            if j == 0:
                # The swap enters this tick at the current price (mid-tick), so
                # containment uses the capacity actually traversed (current price
                # down to the lower boundary), not the full tick width. Using the
                # full width underestimates L0 and lets the optimizer pick a
                # liquidity below containment, where the closed-form model (which
                # assumes the swap stays in the tick) diverges from simulation.
                L0 = (net_total / traversed_cap - P) if traversed_cap > 0 else 0.0
            else:
                Dm = dx - P * cap_per_L
                cap_prev = out[j - 1].cap_per_L
                L0 = (Dm / cap_prev) if cap_prev > 0 else 0.0
            L0 = max(0.0, L0)

            out.append(TickParams(lower, upper, P, dx, R, C, A,
                                  L_inner, L0, L_max, cap_per_L, traversed_cap,
                                  fc_slope))
        return out

    def _solve(self, params: list[TickParams], F, px) -> dict:
        """Pick the best position, ranking candidates by the closed-form utility.

        Returns the position only ({lower_tick, upper_tick, liquidity}); the
        closed-form utility is used solely for internal ranking here.
        """
        # A rational JIT LP always has the option to add nothing (L=0, utility 0),
        # so 0 is the floor: never report a position whose utility is negative
        # (the paper's "Point 1" caveat). This mirrors the combinatorial optimizer,
        # whose line search includes L=0.
        best_position = self._empty()
        best_score = 0.0
        for j, p in enumerate(params):
            if j == 0:
                target, L = p, self._lemma_5_1(p, F)
            else:
                prev = params[j - 1]
                L_low, L_up = self._lemma_5_2(p, prev, F, px)
                target, L = (prev, L_up) if L_up > 0 else (p, L_low)

            score = self._range_utility(target, L, F, px)
            if score > best_score:
                best_score = score
                best_position = {"lower_tick": target.lower,
                                 "upper_tick": target.upper, "liquidity": L}
        return best_position

    # ------------------------------------------------------------------ #
    #  Lemmas (liquidity is always clamped to the budget cap L_max)      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _lemma_5_1(p: TickParams, F) -> float:
        """Optimal liquidity for the innermost range (j=0)."""
        R, P, C = p.R, p.P, p.C
        if R > F and P >= F * C / (R - F):
            L = p.L0                                    # (a)
        elif F > R and P >= R * C / (F - R):
            L = p.L_max                                 # (b)
        else:
            L = max(p.L0, min(p.L_inner, p.L_max))      # (c)
        return min(max(0.0, L), p.L_max)

    @classmethod
    def _lemma_5_2(cls, p: TickParams, prev: TickParams, F, px):
        """
        Optimal liquidity for outer ranges (j>0).

        Returns (L_low, L_up): liquidity in the current range, or (regime
        (d)-upper) in the previous range instead.
        """
        R, P, C = p.R, p.P, p.C
        ratio = math.sqrt(prev.R / R) if (prev.R > 0 and R > 0) else 1.0
        L_up = 0.0

        if R > F and P >= F * C / (R - F):
            L_low = 0.0                                       # (a)
        elif R < F and F < R * ratio and P >= R * C / (F - R):
            L_low = p.L_max                                   # (b)
        elif (R < F and F < R * ratio and P < R * C / (F - R)) or \
             (R > F and P < F * C / (R - F)):
            L_low = cls._lemma_5_1(p, F)                       # (c)
        else:                                                 # (d)
            L0_up = min(prev.L0, prev.L_max)
            U_up = cls._range_utility(prev, L0_up, F, px)
            L_cand = cls._lemma_5_1(p, F)
            U_low = cls._range_utility(p, L_cand, F, px)
            if U_up > U_low:
                L_low, L_up = 0.0, L0_up
            else:
                L_low = L_cand
        return min(max(0.0, L_low), p.L_max), L_up

    @classmethod
    def _range_utility(cls, p: TickParams, L, F, px) -> float:
        """Utility of liquidity L in range p, valid across BOTH regimes.

        If L contains the swap (L >= L0) the tick is terminal and the concave
        contained-utility applies. Otherwise the swap crosses the tick and only
        the linear fully-crossed utility (over the traversed slice) is earned.
        The two coincide at L = L0. Using the contained formula when L < L0
        overvalues ranges the budget cannot hold the swap in -- the defect that
        made the analytical rank an infeasible range above the true optimum.
        """
        if L <= 0:
            return 0.0
        if L >= p.L0:
            return cls._utility(L, p.P, p.dx, p.R, p.C, F, px)
        return max(0.0, L * p.fc_slope)

    @staticmethod
    def _utility(L, P, dx, R, C, F, px, psi=1.0):
        """Closed-form utility (MATLAB utility.m): JIT LP's fees + price impact."""
        if L <= 0:
            return 0.0
        total = L + P
        if total == 0 or (C + total) == 0:
            return 0.0
        const = px * (L / total) * dx
        const_psi = px * (L**psi / (L**psi + P**psi)) * dx
        fees = const_psi * (F - 1)
        return fees + const_psi - const * R * total / (C + total)

    @staticmethod
    def _empty() -> dict:
        return {"lower_tick": None, "upper_tick": None, "liquidity": None}
