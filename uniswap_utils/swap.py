from decimal import Decimal, getcontext
from uniswap_utils.position import Position
from uniswap_utils.utils import (
    tick_from_sqrt_price,
    calculate_active_liquidity,
    get_all_ticks, 
    get_next_tick,
    print_debug
)

class Swap:
    def __init__(self, amount0,amount1, start_price, passive_dict, tick_space, fee_rate, dec0, dec1):
        self.amount0 = amount0  
        self.amount1 = amount1  
        self.zeroForOne = amount0 > 0 
        self.start_price = start_price
        self.passive_dict = passive_dict
        self.tick_space = tick_space
        self.fee_rate = fee_rate
        self.dec0 = dec0
        self.dec1 = dec1


    def zeroForOneSwap(self, remaining_in, current_sqrtP, target_sqrtP, L):
        
        fee_rate = Decimal(self.fee_rate)

        gross_in = remaining_in / (1 + fee_rate)
        feeAmount = remaining_in - gross_in
        available_in = L * (1 / target_sqrtP - 1 / current_sqrtP)
        if available_in > gross_in:
            actual_in = gross_in
            inv_future_sqrtP = 1 / current_sqrtP + actual_in / L
            future_sqrtP = 1 / inv_future_sqrtP
            remaining_in = Decimal("0")
            next_tick = tick_from_sqrt_price(future_sqrtP, self.dec0,self.dec1)
        else:
            actual_in = available_in
            future_sqrtP = target_sqrtP
            feeAmount = actual_in * fee_rate
            remaining_in = remaining_in - (actual_in * (1 + fee_rate))
            next_tick = tick_from_sqrt_price(future_sqrtP, self.dec0,self.dec1) - 1
        amount_out = (current_sqrtP - future_sqrtP ) * L
        return remaining_in, future_sqrtP, next_tick, amount_out, feeAmount

    def oneForZeroSwap(self, remaining_in, current_sqrtP, target_sqrtP, L):

        fee_rate = Decimal(self.fee_rate)

        gross_in = remaining_in / (1 + fee_rate)
        feeAmount = remaining_in - gross_in

        available_in = L * (target_sqrtP - current_sqrtP)

        if available_in > gross_in:
            actual_in = gross_in
            future_sqrtP = current_sqrtP + (actual_in) / L
            remaining_in = Decimal("0")
            next_tick = tick_from_sqrt_price(future_sqrtP, self.dec0,self.dec1)
        else:
            actual_in = available_in
            future_sqrtP = target_sqrtP
            feeAmount = actual_in * fee_rate
            remaining_in = remaining_in - (actual_in * (1 + fee_rate))
            next_tick = tick_from_sqrt_price(future_sqrtP, self.dec0,self.dec1) + 1
            
        amount_out = (1 / current_sqrtP - 1 / future_sqrtP) * L
        return remaining_in, future_sqrtP, next_tick, amount_out, feeAmount

    def simulate(
        self,
        position: Position,
    ) -> dict:

        # Set high precision for Decimal operations
        getcontext().prec = 28

        # Initialize state
        remaining_in = Decimal(self.amount0 if self.amount0 > 0 else self.amount1)
        current_sqrt = Decimal(self.start_price)
        current_tick = tick_from_sqrt_price(current_sqrt,self.dec0,self.dec1)

        # Track outputs and fees
        out_amount = Decimal(0)
        fees_passive = Decimal(0)
        fees_jit = Decimal(0)

        jit_liq = position.to_dict(self.tick_space)

        ticks = get_all_ticks(self.passive_dict, jit_liq)

        while remaining_in > 0:
            if self.zeroForOne:
                direction = "down"
                _, target_sqrt = get_next_tick(current_tick, ticks, direction, self.dec0, self.dec1)
                liq_P, liq_J, L = calculate_active_liquidity(current_tick, self.passive_dict, jit_liq, self.tick_space)
                if L == 0:
                    print_debug(f"Liquidity is zero, exiting loop. Current tick:{current_tick}")
                    break
                remaining_in, current_sqrt, current_tick, partial_out, feeAmount = self.zeroForOneSwap(remaining_in, current_sqrt, target_sqrt, L)
            else:
                direction = "up"
                _, target_sqrt = get_next_tick(current_tick, ticks, direction, self.dec0, self.dec1)
                liq_P, liq_J, L = calculate_active_liquidity(current_tick, self.passive_dict, jit_liq, self.tick_space)
                if L == 0:
                    print_debug(f"Liquidity is zero, exiting loop. Current tick:{current_tick}")
                    break
                remaining_in, current_sqrt, current_tick, partial_out, feeAmount = self.oneForZeroSwap(remaining_in, current_sqrt, target_sqrt, L)

            out_amount += partial_out

            fees_passive += liq_P/L * feeAmount
            fees_jit += liq_J/L * feeAmount
        return {
            "final_sqrt_price": float(current_sqrt),
            "final_tick": current_tick,
            "output_amount": float(out_amount),
            "fees_passive_lp": float(fees_passive),
            "fees_jit_lp": float(fees_jit),
        }

