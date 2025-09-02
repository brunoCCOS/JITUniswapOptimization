from decimal import Decimal
from uniswap_utils.utils import sqrt_price_from_tick
import numpy as np

class Position:
    def __init__(self, liq, lower_tick, upper_tick):
        self.lower_tick = lower_tick
        self.upper_tick = upper_tick
        self.liq = liq

    def to_dict(self, tick_space):
        return {tick:self.liq for tick in range(self.lower_tick,self.upper_tick, tick_space)}

    def value(
        self,
        current_sqrtP,
        price0,
        price1,
        dec0,
        dec1
        ):

        # Calculate base budget for providing max_passive_liq across the entire swap range
        amt0, amt1 = self.tokens(current_sqrtP, dec0, dec1)
        value = amt0 * price0 + amt1 * price1
        return value

    def liqudity_from_budget(
        self,
        budget,
        current_sqrtP,
        price0,price1,
        dec0,
        dec1
    ):
        """
        Given a budget and a tick range (lower_tick, upper_tick), compute the liquidity amount.
        If inputs are not Decimals, they are cast to Decimal.
        """
        # Ensure inputs are Decimals
        budget = budget if isinstance(budget, Decimal) else Decimal(str(budget))
        lower_tick = self.lower_tick if isinstance(self.lower_tick, int) else int(self.lower_tick)
        price0 = price0 if isinstance(price0, Decimal) else Decimal(str(price0))
        price1 = price1 if isinstance(price1, Decimal) else Decimal(str(price1))

        upper_tick = self.upper_tick if isinstance(self.upper_tick, int) else int(self.upper_tick)
        current_sqrtP = (
            current_sqrtP
            if isinstance(current_sqrtP, Decimal)
            else Decimal(str(current_sqrtP))
        )
        dec0 = dec0 if isinstance(dec0, int) else int(dec0)
        dec1 = dec1 if isinstance(dec1, int) else int(dec1)

        sqrtP_lower = sqrt_price_from_tick(lower_tick, dec0, dec1)
        sqrtP_upper = sqrt_price_from_tick(upper_tick, dec0, dec1)

        if current_sqrtP <= sqrtP_lower:
            liquidity = budget / ((Decimal(1) / sqrtP_lower - Decimal(1) / sqrtP_upper) * price0)
        elif current_sqrtP >= sqrtP_upper:
            liquidity = budget / ((sqrtP_upper - sqrtP_lower) * price1)
        else:
            liquidity = budget / (((Decimal(1) / current_sqrtP) - (Decimal(1) / sqrtP_upper))*price0 + (current_sqrtP - sqrtP_lower) * price1)

        return np.float64(liquidity)


    def tokens(
        self, current_sqrtP, dec0, dec1
    ):
        """
        Given a liquidity amount and a tick range (lower_tick, upper_tick), compute the amounts of token0 and token1.
        If inputs are not Decimals, they are cast to Decimal.

        If current_sqrtP <= sqrt(P_lower):
             token0 = L * (1/sqrt(P_lower) - 1/sqrt(P_upper))
             token1 = 0
        If current_sqrtP >= sqrt(P_upper):
             token0 = 0
             token1 = L * (sqrt(P_upper) - sqrt(P_lower))
        Otherwise:
             token0 = L * (1/current_sqrtP - 1/sqrt(P_upper))
             token1 = L * (current_sqrtP - sqrt(P_lower))
        """
        # Ensure inputs are Decimals
        liquidity = self.liq if isinstance(self.liq, Decimal) else Decimal(str(self.liq))
        lower_tick = self.lower_tick if isinstance(self.lower_tick, int) else int(self.lower_tick)
        upper_tick = self.upper_tick if isinstance(self.upper_tick, int) else int(self.upper_tick)
        current_sqrtP = (
            current_sqrtP
            if isinstance(current_sqrtP, Decimal)
            else Decimal(str(current_sqrtP))
        )
        dec0 = dec0 if isinstance(dec0, int) else int(dec0)
        dec1 = dec1 if isinstance(dec1, int) else int(dec1)

        sqrtP_lower = sqrt_price_from_tick(lower_tick, dec0, dec1)
        sqrtP_upper = sqrt_price_from_tick(upper_tick, dec0, dec1)

        print("Tchau:",self.liq, liquidity, current_sqrtP, sqrtP_lower, sqrtP_upper)
        if current_sqrtP <= sqrtP_lower:
            token0 = liquidity * ((Decimal(1) / sqrtP_lower) - (Decimal(1) / sqrtP_upper))
            token1 = Decimal(0)
        elif current_sqrtP >= sqrtP_upper:
            token0 = Decimal(0)
            token1 = liquidity * (sqrtP_upper - sqrtP_lower)
        else:
            token0 = liquidity * ((Decimal(1) / current_sqrtP) - (Decimal(1) / sqrtP_upper))
            token1 = liquidity * (current_sqrtP - sqrtP_lower)
        return np.float64(token0), np.float64(token1)

