from uniswap_utils.swap import Swap
from uniswap_utils.position import Position
from uniswap_utils.utils import tick_from_sqrt_price

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
        self.uppert_tick = upper

    def set_liq(self, liq: float):
        self.liq = liq

    def utility_liq(self, liq):
        position = Position(liq, self.lower_tick, self.uppert_tick)
        return self._utility(position)

    def utility_tick(self, lower,upper):
        position = Position(self.liq, lower, upper)
        return self._utility(position)

    def _utility(self, position):
        init_amt0, init_amt1 = position.tokens(self.swap.state.price, self.swap.state.dec0, self.swap.state.dec1)
        init_value = init_amt0 * self.price0 + init_amt1 * self.price1

        sim_max = self.swap.simulate(position)
        
        end_amt0, end_amt1 = position.tokens(sim_max["final_sqrt_price"], self.swap.state.dec0, self.swap.state.dec1)
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
                 opt_func,
                 **func_args
                 ):
        # Find q*
        null_position = Position(0,0,0)
        noJIT = self.swap.simulate(null_position)
        end_tick = noJIT['final_tick']
        start_tick = tick_from_sqrt_price(self.swap.state.price,self.swap.state.dec0,self.swap.state.dec1)

        for a in range (start_tick, end_tick, self.swap.state.tick_space):
            for b in range (a, end_tick, self.swap.state.tick_space):
                opt_func(self.utility_liq, **func_args)


        

