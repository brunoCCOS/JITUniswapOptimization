from decimal import Decimal
from uniswap_utils.position import Position
from uniswap_utils.state import State
from uniswap_utils.utils import tick_from_sqrt_price

class Swap:
    def __init__(self, amount_in, zeroForOne, state):
        self.amount_in = amount_in
        self.zeroForOne = zeroForOne
        self.state = state

    def update_state(self, state: State):
        self.state = state
        return self.state

    def zeroForOneSwap(self, remaining_in, current_sqrtP, target_sqrtP, L, boundary_tick=None):

        fee_rate = Decimal(self.state.fee_rate)

        gross_in = remaining_in / (1 + fee_rate)
        feeAmount = remaining_in - gross_in
        available_in = L * (1 / target_sqrtP - 1 / current_sqrtP)
        if available_in > gross_in:
            actual_in = gross_in
            inv_future_sqrtP = 1 / current_sqrtP + actual_in / L
            future_sqrtP = 1 / inv_future_sqrtP
            remaining_in = Decimal("0")
            next_tick = tick_from_sqrt_price(future_sqrtP, self.state.dec0,self.state.dec1)
        else:
            actual_in = available_in
            future_sqrtP = target_sqrtP
            feeAmount = actual_in * fee_rate
            remaining_in = remaining_in - (actual_in * (1 + fee_rate))
            # boundary_tick is the tick get_next_tick aimed for (= lower_tick).
            # tick_from_sqrt_price(sqrt(boundary_tick)) == boundary_tick - 1 always,
            # so the original formula gave boundary_tick - 1 - 1 = boundary_tick - 2.
            if boundary_tick is not None:
                next_tick = boundary_tick - 2
            else:
                next_tick = tick_from_sqrt_price(future_sqrtP, self.state.dec0,self.state.dec1) - 1
        amount_out = (current_sqrtP - future_sqrtP ) * L
        return remaining_in, future_sqrtP, next_tick, amount_out, feeAmount

    def oneForZeroSwap(self, remaining_in, current_sqrtP, target_sqrtP, L, boundary_tick=None):

        fee_rate = Decimal(self.state.fee_rate)

        gross_in = remaining_in / (1 + fee_rate)
        feeAmount = remaining_in - gross_in

        available_in = L * (target_sqrtP - current_sqrtP)

        if available_in > gross_in:
            actual_in = gross_in
            future_sqrtP = current_sqrtP + (actual_in) / L
            remaining_in = Decimal("0")
            next_tick = tick_from_sqrt_price(future_sqrtP, self.state.dec0,self.state.dec1)
        else:
            actual_in = available_in
            future_sqrtP = target_sqrtP
            feeAmount = actual_in * fee_rate
            remaining_in = remaining_in - (actual_in * (1 + fee_rate))
            # boundary_tick is the tick get_next_tick aimed for (= lower_tick + ts).
            # tick_from_sqrt_price(sqrt(boundary_tick)) == boundary_tick - 1 always,
            # so the original formula gave boundary_tick - 1 + 1 = boundary_tick.
            if boundary_tick is not None:
                next_tick = boundary_tick
            else:
                next_tick = tick_from_sqrt_price(future_sqrtP, self.state.dec0,self.state.dec1) + 1

        amount_out = (1 / current_sqrtP - 1 / future_sqrtP) * L
        return remaining_in, future_sqrtP, next_tick, amount_out, feeAmount

    def simulate(
        self,
        position: Position,
    ) -> dict:
        """
        Simulate a swap using Q96 integer arithmetic (same approach as the enricher).

        Inputs (state.price, passive_dict, amount_in) are in library/human units.
        Internally everything is converted to raw Q96 integers to avoid slow Decimal
        arithmetic.  Outputs are converted back to library units before returning,
        so the external interface is unchanged.
        """
        import math

        Q96 = 1 << 96
        dec0, dec1 = self.state.dec0, self.state.dec1
        ts = self.state.tick_space
        fee_rate = float(self.state.fee_rate)
        log_base = math.log(1.0001)

        # ── Unit conversion helpers ────────────────────────────────────────────
        # Library sqrt price:  lib_sqrt = raw_sqrt_unit / 10^((dec1-dec0)/2)
        # Raw Q96 sqrt price:  raw_sqrt_x96 = sqrt(1.0001^tick) * Q96
        # Relation:            raw_sqrt_x96 = lib_sqrt * 10^((dec1-dec0)/2) * Q96
        sqrt_adj = 10.0 ** ((dec1 - dec0) / 2.0)   # lib → dimensionless
        liq_scale = 10.0 ** ((dec0 + dec1) / 2.0)  # lib L → raw L
        dec_in = dec1 if not self.zeroForOne else dec0

        # passive_raw is constant for a given state — cache it on self so it's only
        # converted once across the many simulate() calls the optimizer makes.
        if getattr(self, '_sim_cache_id', None) != id(self.state):
            self._sim_cache_id = id(self.state)
            self._passive_raw = {
                k: int(float(v) * liq_scale) for k, v in self.state.passive_dict.items()
            }

        # Convert starting price and JIT position to raw integer units
        current_sqrt_x96 = int(float(self.state.price) * sqrt_adj * Q96)
        passive_raw = self._passive_raw
        jit_raw = {k: int(float(v) * liq_scale) for k, v in position.to_dict(ts).items()}

        # Gross amount in (includes fee) in raw integer tokens
        remaining_in = int(float(self.amount_in) * 10 ** dec_in)

        # Current tick from raw sqrt price
        current_tick = int(2.0 * math.log(current_sqrt_x96 / Q96) / log_base)

        fees_passive = 0.0   # raw fee tokens accumulated
        fees_jit = 0.0
        # output_amount is not used by the optimizer (only fees and final price matter),
        # so skip the expensive 200-bit BigInt multiplications that computing it requires.

        # Cache bsqrt computations on the instance: the same boundary ticks are visited
        # on every simulate() call (they're fixed by passive_dict), so after the first
        # call the values are just dict lookups instead of float pow operations.
        if not hasattr(self, '_bsqrt_cache'):
            self._bsqrt_cache: dict[int, int] = {}
        _bc = self._bsqrt_cache

        def _bsqrt(tick: int) -> int:
            v = _bc.get(tick)
            if v is None:
                v = int(1.0001 ** (tick / 2.0) * Q96)
                _bc[tick] = v
            return v

        while remaining_in > 0:
            active_tick = (current_tick // ts) * ts
            L_p = passive_raw.get(active_tick, 0)
            L_j = jit_raw.get(active_tick, 0)
            L = L_p + L_j

            if L == 0:
                print(f"Liquidity is zero, exiting loop. Current tick:{current_tick}")
                break

            # Net input (fee stripped) that actually moves the price
            gross_in = int(remaining_in / (1.0 + fee_rate))
            fee_amount = remaining_in - gross_in

            if not self.zeroForOne:
                # ── Up swap: token1 in, token0 out ────────────────────────────
                # Capacity = L * (sqrt_upper - sqrt_current) / Q96
                boundary_tick = active_tick + ts
                bsqrt = _bsqrt(boundary_tick)
                delta = bsqrt - current_sqrt_x96
                if delta <= 0:
                    current_tick = boundary_tick
                    current_sqrt_x96 = bsqrt
                    continue
                cap = L * delta // Q96
                if gross_in <= cap:
                    # Swap terminates in this range
                    new_sqrt = current_sqrt_x96 + gross_in * Q96 // L
                    fees_passive += fee_amount * L_p / L
                    fees_jit += fee_amount * L_j / L
                    current_sqrt_x96 = new_sqrt
                    remaining_in = 0
                else:
                    # Cross the boundary
                    consumed = int(cap * (1.0 + fee_rate))
                    fee_at_step = consumed - cap
                    fees_passive += fee_at_step * L_p / L
                    fees_jit += fee_at_step * L_j / L
                    remaining_in = max(0, remaining_in - consumed)
                    current_sqrt_x96 = bsqrt
                    current_tick = boundary_tick

            else:
                # ── Down swap: token0 in, token1 out ──────────────────────────
                # Capacity = L * (sqrt_current - sqrt_lower) * Q96 / (sqrt_lower * sqrt_current)
                boundary_tick = active_tick
                bsqrt = _bsqrt(boundary_tick)
                if bsqrt >= current_sqrt_x96:
                    # Already at or below boundary; step one ts down
                    current_tick = boundary_tick - ts
                    current_sqrt_x96 = bsqrt
                    continue
                denom_prod = bsqrt * current_sqrt_x96
                cap = L * (current_sqrt_x96 - bsqrt) * Q96 // denom_prod if denom_prod > 0 else 0
                if cap == 0:
                    current_tick = boundary_tick - ts
                    current_sqrt_x96 = bsqrt
                    continue
                if gross_in <= cap:
                    # Swap terminates in this range
                    # new_sqrt = L * Q96 * current / (gross_in * current + L * Q96)
                    new_sqrt = (L * Q96 * current_sqrt_x96) // (gross_in * current_sqrt_x96 + L * Q96)
                    fees_passive += fee_amount * L_p / L
                    fees_jit += fee_amount * L_j / L
                    current_sqrt_x96 = new_sqrt
                    remaining_in = 0
                else:
                    # Cross the boundary
                    consumed = int(cap * (1.0 + fee_rate))
                    fee_at_step = consumed - cap
                    fees_passive += fee_at_step * L_p / L
                    fees_jit += fee_at_step * L_j / L
                    remaining_in = max(0, remaining_in - consumed)
                    current_sqrt_x96 = bsqrt
                    # After crossing down: new tick is one below this boundary's range
                    current_tick = boundary_tick - 2

        # ── Convert outputs back to library/human units ────────────────────────
        # raw_sqrt_x96 = lib_sqrt * sqrt_adj * Q96 → lib_sqrt = raw / (sqrt_adj * Q96)
        final_sqrt_human = (current_sqrt_x96 / Q96) / sqrt_adj
        final_tick = int(2.0 * math.log(max(current_sqrt_x96, 1) / Q96) / log_base)

        return {
            "final_sqrt_price": final_sqrt_human,
            "final_tick": final_tick,
            "output_amount": 0.0,
            "fees_passive_lp": fees_passive / 10 ** dec_in,
            "fees_jit_lp": fees_jit / 10 ** dec_in,
        }
