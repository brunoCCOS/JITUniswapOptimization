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
from decimal import Decimal

from uniswap_utils.swap import Swap
from uniswap_utils.position import Position
from uniswap_utils.utils import (
    get_tick_from_sqrt_price,
    get_sqrt_price_from_tick,
    move_tick_idx
)


# TODO: The current implementation only accounts for zeroForOne swaps. It does not implement the opposite direction

@dataclass
class TickParams:
    """Per-range parameters feeding the closed-form solution."""
    lower_idx: int 
    upper_idx: int
    lower: float
    upper: float
    lower_sqrt: float
    upper_sqrt: float
    q_hat: float
    q_hat_sqrt: float
    delta_inv_sqrt_full_range: float
    delta_inv_sqrt_part_range: float
    delta_sqrt_full_range: float
    delta_sqrt_part_range: float
    dx: float
    volumes_tokens0_in_range: float
    traversed_tokens0: float
    P: float
    R: float
    R_prime: float
    C: float
    A: float
    epsilon: float
    L_max: float
    L_min: float
    L_max_budget_dollars: float
    L_inner: float
    L_bar: float
    L_star: float
    L_B: float

class AnalyticalOptimizer:
    """Closed-form JIT liquidity optimizer (Lemmas 5.1 / 5.2)."""

    def __init__(self, swap: Swap, price0: float, price1: float):
        self.swap = swap
        self.price0 = price0
        self.price1 = price1

    # ------------------------------------------------------------------ #

    def optimize(self, budget) -> dict:
        """Return {lower_tick, upper_tick, liquidity, utility} for the best range. `budget` is expressed in dollars."""
        state = self.swap.state
        ts = state.tick_space
        tick_idx_offset = state.tick_idx_offset
        dec0, dec1 = state.dec0, state.dec1
        F = 1.0 + float(state.fee_rate)
        Delta_x = float(self.swap.amount_in)
        direction_up = not self.swap.zeroForOne

        # Reframe into canonical (downward) coordinates. px/py are the USD prices
        # of the input/output tokens; canon_sqrt maps a tick to its sqrt price in
        # the canonical frame (identity for a down swap, inverted for an up swap).
        pool_sqrt = float(state.price_sqrt)
        if direction_up:
            
            px, py = float(self.price1), float(self.price0)  # in=token1, out=token0
            init_sqrt = 1.0 / pool_sqrt
            canon_sqrt = lambda t: 1.0 / float(
                self._get_tick_sqrt_price_from_tick_idx(t, state.tick_idx_offset, state.tick_space, dec0, dec1)
            )
            canon_tick_idx = lambda p: self._get_tick_idx_from_tick_sqrt_price(
                1 / p, state.tick_idx_offset, state.tick_space, dec0, dec1
            )
            canon_get_next_tick_idx = lambda t: self._move_tick_idx(t, 1, state.tick_space)
        else:
            px, py = float(self.price0), float(self.price1)  # in=token0, out=token1
            init_sqrt = pool_sqrt
            canon_sqrt = lambda t: float(
                self._get_tick_sqrt_price_from_tick_idx(t, state.tick_idx_offset, state.tick_space, dec0, dec1)
            )
            canon_tick_idx = lambda p: self._get_tick_idx_from_tick_sqrt_price(
                p, state.tick_idx_offset, state.tick_space, dec0, dec1
            )
            canon_get_next_tick_idx = lambda t: self._move_tick_idx(t, -1, state.tick_space)

        start_tick = canon_tick_idx(init_sqrt)

        _, end_range, remaining = self.simulate_swap(
            Decimal(init_sqrt), state.passive_dict, 
            {}, Delta_x, tick_idx_offset, ts, dec0, dec1,
            canon_sqrt, canon_tick_idx, canon_get_next_tick_idx, direction_up
        )
        end_tick = end_range[0]

        ranges = self._build_ranges(start_tick, end_tick, ts, direction_up)
        
        if not ranges:
            # The swap stays within the current tick-space range (it does not
            # cross a range boundary). Lemma 5.1 still applies to that single
            # range and can prescribe positive liquidity, so fall back to it
            # instead of returning "no JIT". (Gating on integer-tick equality
            # wrongly skipped this whole class of swaps and made the result
            # depend on an integer-tick crossing irrelevant to the tick-space
            # ranges actually optimized over.)
            ranges = [(start_tick, self._move_tick_idx(start_tick, 1, ts))]

        params = self._precompute(
            ranges, Delta_x, budget, px, py, F, init_sqrt, canon_sqrt
        )
        return self._solve(params, F, budget, px)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _move_tick_idx(i, delta, tick_space):
        return move_tick_idx(i, delta, relative = False, tick_space = tick_space)

    @staticmethod
    def _get_tick_idx_from_tick_sqrt_price(tick_sqrt_price, tick_idx_offset, tick_space, dec0, dec1):
        return get_tick_from_sqrt_price(
            Decimal(tick_sqrt_price), dec0, dec1, base = Decimal("1.0001"),
            relative = False, offset = tick_idx_offset, tick_space = tick_space, human = True
        )

    @staticmethod
    def _get_tick_sqrt_price_from_tick_idx(tick_idx, tick_idx_offset, tick_space, dec0, dec1):
        sqrt_price = get_sqrt_price_from_tick(
            tick_idx, dec0, dec1, base = Decimal("1.0001"),
            relative = False, offset = tick_idx_offset, tick_space = tick_space, human = True
        )
        return sqrt_price

    @classmethod
    def simulate_swap(
        cls, pool_sqrt_price: Decimal, passive_dict, positions, 
        amount_in, tick_idx_offset, tick_space, dec0, dec1, 
        canon_sqrt, canon_tick_idx, canon_get_next_tick_idx, direction_up
    ) -> None | tuple[float, tuple[int, int], float]:
        pool_sqrt_price = Decimal(pool_sqrt_price)
        rem = Decimal(amount_in)

        pool_price_tick_idx = canon_tick_idx(pool_sqrt_price)
        next_tick_idx = canon_get_next_tick_idx(pool_price_tick_idx)
        next_tick_sqrt = Decimal(canon_sqrt(next_tick_idx))

        if direction_up:
            lower_tick_idx, upper_tick_idx = pool_price_tick_idx, next_tick_idx
            lower_tick_sqrt, upper_tick_sqrt = next_tick_sqrt, pool_sqrt_price
        else:
            lower_tick_idx, upper_tick_idx = next_tick_idx, pool_price_tick_idx
            lower_tick_sqrt, upper_tick_sqrt = next_tick_sqrt, pool_sqrt_price

        non_zero_liquidity_tick_idxs = set(
            [tick_idx for tick_idx, liq in passive_dict.items() if liq > 0]
        ).union(
            set([tick_idx for tick_idx, liq in positions.items() if liq > 0])
        )
        min_tick_idx_non_empty_liquidity = min(non_zero_liquidity_tick_idxs)
        max_tick_idx_non_empty_liquidity = max(non_zero_liquidity_tick_idxs)

        while rem > 0:
            consumed = 0
            K = Decimal(passive_dict.get(lower_tick_idx, 0) + positions.get(lower_tick_idx, 0))

            if K > 0:
                consumed = K * (Decimal(1) / lower_tick_sqrt - Decimal(1) / upper_tick_sqrt)
                if rem <= consumed:
                    final_pool_sqrt_price = upper_tick_sqrt / (Decimal(1) + Decimal(rem) * upper_tick_sqrt / K)
                    final_tick_lower_idx = canon_tick_idx(final_pool_sqrt_price)
                    final_tick_upper_idx = canon_get_next_tick_idx(final_tick_lower_idx)
                    return final_pool_sqrt_price, (final_tick_lower_idx, final_tick_upper_idx), 0
                rem -= consumed

            upper_tick_sqrt = lower_tick_sqrt
            lower_tick_idx = canon_get_next_tick_idx(lower_tick_idx)
            lower_tick_sqrt = Decimal(canon_sqrt(lower_tick_idx))
                
            # This accounts for both down and up directions when there are no longer positive liquidities allocated in the next ranges
            if lower_tick_idx < min_tick_idx_non_empty_liquidity or lower_tick_idx > max_tick_idx_non_empty_liquidity:
                return None

    @classmethod
    def _build_ranges(cls, start_tick, end_tick, ts, direction_up = False):
        """Ranges from the initial price (j=0) toward the final price."""
        ranges = []
        if direction_up:
            lo = start_tick
            while lo <= end_tick:
                hi = cls._move_tick_idx(lo, 1, ts)
                ranges.append((lo, hi))
                lo = hi
        else:
            hi = cls._move_tick_idx(start_tick, 1, ts)
            while hi > end_tick:
                lo = cls._move_tick_idx(hi, -1, ts)
                ranges.append((lo, hi))
                hi = lo
        return ranges

    def _precompute(self, ranges, Delta_x, budget_dollars, px, py, F, init_sqrt_price, canon_sqrt) -> list[TickParams]:        
        # net_total = Delta_x / F     # TODO: Ask what we should do regarding the net_total. Should we ignore the fees here? If so, we will likely need to also update the combinatorial method. 
        net_total = Delta_x

        init_sqrt_price = float(init_sqrt_price)

        out: list[TickParams] = []
        for j, (lower_idx, upper_idx) in enumerate(ranges):
            # lower_sqrt, upper_sqrt = float(sqrt_price_from_tick(lower_idx, dec0, dec1)), float(sqrt_price_from_tick(upper_idx, dec0, dec1))
            # lower, upper = math.pow(lower_sqrt, 2), math.pow(upper_sqrt, 2)
            a, b = canon_sqrt(lower_idx), canon_sqrt(upper_idx)
            lower_sqrt, upper_sqrt = min(a, b), max(a, b)
            lower, upper = lower_sqrt ** 2, upper_sqrt ** 2

            q_hat_sqrt = min(init_sqrt_price, upper_sqrt) # TODO: Check if this will change depending on the swap direction 
            q_hat = q_hat_sqrt ** 2

            delta_inv_sqrt_full_range = (1/lower_sqrt - 1/upper_sqrt)
            delta_inv_sqrt_part_range = (1/lower_sqrt - 1/q_hat_sqrt)

            delta_sqrt_full_range = (upper_sqrt - lower_sqrt)
            delta_sqrt_part_range = (q_hat_sqrt - lower_sqrt)

            P = float(self.swap.state.passive_dict.get(lower_idx, 0.0))

            dx = net_total if j == 0 else out[j-1].dx - out[j-1].traversed_tokens0

            volumes_tokens0_in_range = P * delta_inv_sqrt_full_range
            traversed_tokens0 = min(P * delta_inv_sqrt_part_range, dx)

            R = q_hat * py / px # Eq. (34)
            C = dx * q_hat_sqrt # Eq. (34)
            A = math.sqrt((F / R) * (P / (C + P)))
            R_prime = None if j == 0 else R * math.sqrt(out[j-1].q_hat / out[j-1].lower)

            epsilon = delta_sqrt_full_range * py

            L_max = dx / delta_inv_sqrt_part_range - P  # Eq. (32)
            L_min = max(0, L_max)  # Eq. (37)
            
            L_max_budget_dollars = budget_dollars / epsilon
            
            L_inner = (C * A) / (1.0 - A) - P          

            L_bar = min(L_max, L_max_budget_dollars)

            L_star = None
            if F > R:
                L_star = min(max(L_inner, L_min), L_max_budget_dollars) if P < (R * C / (F - R)) else L_max_budget_dollars
            
            out.append(TickParams(
                lower_idx = lower_idx, upper_idx = upper_idx, 
                lower = lower, upper = upper, 
                lower_sqrt = lower_sqrt, upper_sqrt = upper_sqrt, 
                q_hat = q_hat,
                q_hat_sqrt = q_hat_sqrt,
                delta_inv_sqrt_full_range = delta_inv_sqrt_full_range,
                delta_inv_sqrt_part_range = delta_inv_sqrt_part_range,
                delta_sqrt_full_range = delta_sqrt_full_range,
                delta_sqrt_part_range = delta_sqrt_part_range,
                dx = dx,
                volumes_tokens0_in_range = volumes_tokens0_in_range,
                traversed_tokens0 = traversed_tokens0,
                P = P,
                R = R,
                R_prime = R_prime,
                C = C,
                A = A,
                epsilon = epsilon,
                L_max = L_max,
                L_min = L_min,
                L_max_budget_dollars = L_max_budget_dollars,
                L_inner = L_inner,
                L_bar = L_bar,
                L_star = L_star,
                L_B = None
            ))

        for j, params in enumerate(out[:-1]):
            params_prev = out[j + 1]

            L_B_num = (
                (params_prev.P + params_prev.L_max_budget_dollars)
                * params_prev.delta_inv_sqrt_full_range
                + params.P * params.delta_inv_sqrt_part_range - params.dx
            )

            sqrt_lower_prev = params_prev.lower_sqrt
            sqrt_lower = params.lower_sqrt
            sqrt_q_hat = params.q_hat_sqrt
            
            L_B_den = (
                (1/sqrt_lower_prev - 1/sqrt_q_hat)
                * (sqrt_q_hat - sqrt_lower) / sqrt_lower
            )

            params.L_B = L_B_num / L_B_den
        return out

    def _solve(self, params: list[TickParams], F, budget, px) -> dict:
        """Pick the best positions, ranking candidates by the closed-form utility.

        Returns the consecutive positions only ({t_{m - 1}, t_{m}, L_{m - 1}}, {t_{m}, t_{m + 1}, L_{m}}); 
        """
        # A rational JIT LP always has the option to add nothing (L=0, utility 0),
        # so 0 is the floor: never report a position whose utility is negative
        # (the paper's "Point 1" caveat). This mirrors the combinatorial optimizer,
        # whose line search includes L=0.
        best_positions = self._empty()
        best_utility = 0.0

        num_ranges = len(params)

        # m_0 = \bar{m} (Lines 2 and 3 in Algorithm 1)
        if num_ranges == 1:
            p = params[0]
            L_m = self._proposition_3_1(p, F, L_max = 0)
            return [{"lower_tick": p.lower_idx, "upper_tick": p.upper_idx, "liquidity": L_m}]
    
        for j in range(num_ranges-1):
            p, p_prev = params[j], params[j+1]
            
            allocations, utility_value = self._theorem_5_3(p, p_prev, F, budget, px)
            if allocations is None:
                continue

            L_prev, L_m = allocations

            if utility_value > best_utility:
                best_positions = [
                    {"lower_tick": p_prev.lower_idx, "upper_tick": p_prev.upper_idx, "liquidity": L_prev},
                    {"lower_tick": p.lower_idx, "upper_tick": p.upper_idx, "liquidity": L_m},
                ]
                best_utility = utility_value
            
        return best_positions

    # ------------------------------------------------------------------ #
    #  Lemmas (liquidity is always clamped to the budget cap L_max)      #
    # ------------------------------------------------------------------ #

    @classmethod
    def _proposition_3_1(cls, p: TickParams, F, L_max = 0) -> float:
        # Single tick range solution
        # Also used within Lemma 5.2

        P = p.P
        R = p.R
        C = p.C

        # F = R
        if math.isclose(F, R):
            print("Proposition 3.1 (F = R)")
            return min(
                math.sqrt(P * (C + P)),
                p.L_max_budget_dollars
            )

        # Case (a):
        if R > F and P >= F * C / (R - F):
            print("Proposition 3.1 (a)")
            return L_max

        # Case (b): F > R and P >= RC / (F - R)
        if F > R and P >= R * C / (F - R):
            print("Proposition 3.1 (b)")
            return p.L_max_budget_dollars

        # Case (c):
        # F > R and P < RC/(F-R), or
        # R > F and P < FC/(R-F)
        print("Proposition 3.1 (c)")
        return max(L_max, min(p.L_inner, p.L_max_budget_dollars))

    @classmethod
    def _lemma_5_1(cls, p: TickParams, p_prev: TickParams, F, B, px) -> None | tuple[float, float]:
        # General tick ranges solution (P1_m)
        
        L_prev_min = max(0, p_prev.L_max)  # Eq. (37), and is equivalent to max(0, p_prev.L_max) (see Eqs. (30)-(32))

        feasible = L_prev_min <= p_prev.L_max_budget_dollars

        if not feasible:
            print("No feasible solution can be provided by Lemma 5.1")
            return None

        R0, C0, P0,= p_prev.R, p_prev.C, p_prev.P
        
        # Caso (c)
        if math.isclose(R0, F):
            print("Candidate solution is given by Lemma 5.1 (c)")
            solution_L_prev = min(
                max(math.sqrt(P0 * (C0 + P0)), L_prev_min),
                p_prev.L_max_budget_dollars
            )
            return (solution_L_prev, 0)

        if R0 > F:
            threshold = F * C0 / (R0 - F)

            # Case (a)
            if P0 >= threshold:
                print("Candidate solution is given by Lemma 5.1 (a)")
                return (L_prev_min, 0)

            # Case (b)
            print("Candidate solution is given by Lemma 5.1 (b)")
            return (
                min(
                    max(p_prev.L_inner, p_prev.L_min), 
                    p_prev.L_max_budget_dollars
                ), 0
            )

        # From here, R0 < F
        R0_prime = p_prev.R_prime
        threshold = R0 * C0 / (F - R0)

        if F <= R0_prime:
            # Case (d)
            if P0 >= threshold:
                print("Candidate solution is given by Lemma 5.1 (d)")
                return (p_prev.L_max_budget_dollars, 0)

            # Case (e)
            print("Candidate solution is given by Lemma 5.1 (e)")
            return (
                min(
                    max(p_prev.L_inner, p_prev.L_min), 
                    p_prev.L_max_budget_dollars
                ), 0
            )

        # Case (f)
        epsilon = p.epsilon
        epsilon_prev = p_prev.epsilon
        
        L_bar = p.L_bar
        L_prev_star = p_prev.L_star

        L_B = p.L_B

        L_prev_R, L_R = (0, L_bar) if L_B >= L_bar else (
            (B - epsilon * L_B) / epsilon_prev, L_B
        )

        first_candidate_solution = (L_prev_star, 0)
        second_candidate_solution = (L_prev_R, L_R)

        first_candidate_utility = cls._range_utility(
            F, px, p, first_candidate_solution[1], p_prev, first_candidate_solution[0]
        )

        second_candidate_utility = cls._range_utility(
            F, px, p, second_candidate_solution[1], p_prev, second_candidate_solution[0]
        )

        if first_candidate_utility >= second_candidate_utility:
            print("Candidate solution is given by the first solution of Lemma 5.1 (f)")
            return first_candidate_solution

        print(first_candidate_solution, first_candidate_utility)
        print(second_candidate_solution, second_candidate_utility)
        print("Candidate solution is given by the second solution of Lemma 5.1 (f)")
        return second_candidate_solution

    @classmethod
    def _lemma_5_2(cls, p: TickParams, F) -> None | tuple[float, float]:
        # General tick ranges solution (P2_m)

        print("Candidate solution is given by Lemma 5.2. Calling Proposition 3.1...")

        L_max = p.L_max
        feasible = L_max <= p.L_max_budget_dollars

        if not feasible:
            print("No feasible solution can be provided by Lemma 5.2")
            return None

        # L*_m from Proposition 3.1 for tick range m, according to Lemma 5.2
        L_star = cls._proposition_3_1(p, F, 0)
        L_tilde = max(L_max, L_star)

        return (0.0, L_tilde)

    @classmethod
    def _theorem_5_3(cls, p: TickParams, p_prev: TickParams, F, B, px):
        solution_lemma_5_1 = cls._lemma_5_1(p, p_prev, F, B, px)
        solution_lemma_5_2 = cls._lemma_5_2(p, F)

        utility_solution_lemma_5_1 = -math.inf if solution_lemma_5_1 is None else cls._range_utility(
            F, px, p, solution_lemma_5_1[1], p_prev, solution_lemma_5_1[0]
        )

        utility_solution_lemma_5_2 = -math.inf if solution_lemma_5_2 is None else cls._range_utility(
            F, px, p, solution_lemma_5_2[1], p_prev, solution_lemma_5_2[0]
        )

        # Case (o)
        if solution_lemma_5_1 is None and solution_lemma_5_2 is None:
            print("No feasible solution can be given by Theorem 5.3 (o)")
            return (None, -math.inf)

        # Case (i)
        if solution_lemma_5_1 is not None and solution_lemma_5_2 is None:
            print("Candidate solution is given by Theorem 5.3 (i)")
            return (solution_lemma_5_1, utility_solution_lemma_5_1)

        if solution_lemma_5_2 is not None:
            # Case (ii)
            if math.pow(p.A, 2) * p.q_hat <= p.lower:
                print("Candidate solution is given by Lemma 5.1, according to Theorem 5.3 (ii)")
                return (solution_lemma_5_1, utility_solution_lemma_5_1)

            # Case (iii)
            else: 
                if utility_solution_lemma_5_1 >= utility_solution_lemma_5_2:
                    print("Candidate solution is given by Lemma 5.1, according to Theorem 5.3 (iii)")
                    return (solution_lemma_5_1, utility_solution_lemma_5_1)
                else:
                    print("Candidate solution is given by Lemma 5.2, according to Theorem 5.3 (iii)")
                    return (solution_lemma_5_2, utility_solution_lemma_5_2)

        raise NotImplementedError(
            "None of the cases covered by Theorem 5.3 could be applied for the input parameters."
        )

    @classmethod
    def _range_utility(cls, F, px, p: TickParams, L_m, p_prev: TickParams = None, L_prev = None) -> float:
        """Utility of liquidity L in range p, valid across BOTH regimes.

        If L contains the swap (L >= L0) the tick is terminal and the concave
        contained-utility applies. Otherwise the swap crosses the tick and only
        the linear fully-crossed utility (over the traversed slice) is earned.
        The two coincide at L = L0. Using the contained formula when L < L0
        overvalues ranges the budget cannot hold the swap in -- the defect that
        made the analytical rank an infeasible range above the true optimum.
        """
        
        q_hat, lower, dx, P, C, R = p.q_hat, p.lower, p.dx, p.P, p.C, p.R

        if L_m > p.L_max:
            # Utility function in the sentence below Eq. (33)
            return px * dx * (F - R * (L_m + P)/(C + L_m + P)) * L_m / (L_m + P)
        
        R0, R0_prime, C0, P0 = p_prev.R, p_prev.R_prime, p_prev.C, p_prev.P

        # Eq. (33)
        range_m_term = px * L_m * (F - R0_prime) * (math.sqrt(p.q_hat) - math.sqrt(p.lower)) / math.sqrt(lower * q_hat)

        dx_prev = dx - (L_m + P) * p.delta_inv_sqrt_part_range # Eq. (32)

        range_prev_term = px * dx_prev * (F - R0 * (L_prev + P0)/(C0 + L_prev + P0)) * L_prev / (L_prev + P0)
        
        return range_m_term + range_prev_term

    @staticmethod
    def _empty():
        return []
