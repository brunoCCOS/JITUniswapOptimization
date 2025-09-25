from uniswap_utils.swap import Swap
from uniswap_utils.position import Position
from uniswap_utils.utils import tick_from_sqrt_price, get_rounded_tick

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
        """
        Find optimal position parameters (ticks and liquidity) within a budget.
        
        Args:
            budget: Maximum amount to use for liquidity provision
            opt_func: Search function from search.py to optimize liquidity
            **func_args: Additional arguments for the search function
            
        Returns:
            Dictionary with optimal parameters (lower_tick, upper_tick, liquidity, utility)
        """
        # Find q*
        null_position = Position(0, 0, 0)
        noJIT = self.swap.simulate(null_position)
        end_tick = noJIT['final_tick']
        current_tick = tick_from_sqrt_price(self.swap.state.price, self.swap.state.dec0, self.swap.state.dec1)
        start_tick, _ = get_rounded_tick(current_tick, self.swap.state.tick_space)
        
        best_utility = float('-inf')
        best_config = {"lower_tick": None, "upper_tick": None, "liquidity": None, "utility": None}
        
        # Iterate through all possible tick ranges
        for a in range(start_tick, end_tick, self.swap.state.tick_space):
            for b in range(a + self.swap.state.tick_space, end_tick + self.swap.state.tick_space, self.swap.state.tick_space):
                # Set current tick range to evaluate
                self.set_ticks(a, b)
                
                # Create a position with this budget and tick range
                position = Position(0, a, b)
                # Calculate max liquidity possible with budget
                max_liq = position.liqudity_from_budget(
                    budget,
                    self.swap.state.price,
                    self.price0,
                    self.price1,
                    self.swap.state.dec0,
                    self.swap.state.dec1
                )
                
                # Define bounds for liquidity search (0 to max_liq)
                liq_bounds = (0, max_liq)
                
                # Use provided search function to find optimal liquidity
                optimal_utility, optimal_liq = opt_func(self.utility_liq, *liq_bounds, **func_args)
                
                # Update best configuration if current is better
                if optimal_utility > best_utility:
                    best_utility = optimal_utility
                    best_config = {
                        "lower_tick": a,
                        "upper_tick": b,
                        "liquidity": optimal_liq,
                        "utility": optimal_utility
                    }
        
        return best_config


            

