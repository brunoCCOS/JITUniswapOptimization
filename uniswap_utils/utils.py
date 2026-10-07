from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_EVEN
from functools import lru_cache
from uniswap_utils import Numerical
import os

def print_debug(msg: str):
    if os.getenv("DEBUG") == "1":
        print(msg)

def move_tick_idx(i, delta, relative: bool = False, tick_space: int = 1):
    if relative: 
        return i + delta
    else:
        return i + delta * tick_space

def get_sqrt_price_from_tick(
    i: int,
    dec0: int,
    dec1: int,
    *,
    base: Decimal = Decimal("1.0001"),
    relative: bool = False,
    offset: int = 0,
    tick_space: int = 1,
    human: bool = True
) -> Decimal:
    """
    Compute the sqrt price associated with a tick index.

    The tick value is derived from the index according to the selected
    indexing scheme.

    Absolute mode:
        tick_idx = offset + floor(i / tick_space) * tick_space

    Relative mode:
        tick_idx = offset + tick_space * i

    The price associated with the tick is:
        price = base ** tick_idx

    When `human=True`, token decimal scaling is applied:
        price = base ** tick_idx * 10 ** (dec0 - dec1)

    Returns:
        The square root of the price associated with the tick.
    """

    if tick_space <= 0:
        raise ValueError("tick_space must be positive")

    if relative:
        tick = offset + tick_space * i
    else:
        tick = offset + (i // tick_space) * tick_space

    tick = Decimal(tick)
    base = Decimal(str(base))

    sqrt_price = base ** (tick / Decimal(2))

    if human:
        decimal_adjustment = Decimal(10) ** (
            Decimal(dec0 - dec1) / Decimal(2)
        )
        sqrt_price *= decimal_adjustment

    return sqrt_price

def get_tick_from_sqrt_price(
    sqrt_price: Decimal,
    dec0: int,
    dec1: int,
    *,
    base: Decimal = Decimal("1.0001"),
    relative: bool = False,
    offset: int = 0,
    tick_space: int = 1,
    human: bool = True,
) -> int:
    """
    Convert a sqrt price to its corresponding tick index.

    When relative=True, rounds down to the nearest tick_space.
    When human=True, adjusts for the token decimal difference.
    """
    human_delta = Decimal(dec1 - dec0) if human else Decimal(0)

    transformed_tick_idx = (Decimal(2) * sqrt_price.log10() + human_delta) / base.log10()

    if relative:
        tick = ((transformed_tick_idx - Decimal(offset)) / Decimal(tick_space)).to_integral_value(
            rounding = ROUND_HALF_EVEN
        )
    else:
        tick = (transformed_tick_idx - Decimal(offset)).to_integral_value(rounding = ROUND_HALF_EVEN)
        tick = (tick // tick_space) * tick_space

    return int(tick)

# Just for retrocompatibility with previous implementations.
@lru_cache(maxsize=131072)
def sqrt_price_from_tick(tick: int,
                         dec0: int,
                         dec1: int):
    """
    Compute actual sqrt(price) from a given tick index, accounting for token decimals.
        sqrt(P) = 1.0001^(tick/2) / 10^((dec1 - dec0)/2)
    where P = price of token1 in terms of token0.
    """
    sqrt_price = get_sqrt_price_from_tick(
        tick, dec0, dec1, 
        base = Decimal("1.0001"),
        relative = False, 
        offset = 0, 
        tick_space = 1, 
        human = True
    )
    
    return sqrt_price


def tick_from_sqrt_price(sqrt_price: Numerical, dec0:int, dec1:int):
    """
    Compute the tick index (rounded down) from an actual sqrt(price), accounting for token decimals.
    Uses a binary search to find the highest tick such that:
         (1.0001^(tick/2)) / factor <= sqrt_price,
    where factor = 10^((dec0-dec1)/2)
    """
    base = Decimal("1.0001") ** Decimal("0.5")
    diff = (Decimal(dec0) - Decimal(dec1)) / Decimal(2)
    factor = Decimal(10) ** diff
    normalized = Decimal(sqrt_price) / factor
    # Uniswap V3 tick bounds
    min_tick, max_tick = -887272, 887272
    floor_tick = min_tick
    while min_tick <= max_tick:
        mid = (min_tick + max_tick) // 2
        mid_val = base ** Decimal(mid)
        if mid_val <= normalized:
            floor_tick = mid
            min_tick = mid + 1
        else:
            max_tick = mid - 1
    return floor_tick


def calculate_active_liquidity(
        current_tick: int,
        passive_liq: dict[int, Numerical],
        jit_liq: dict[int, Numerical],
        tick_space: int):
    """
    Calculate the active liquidity at the current tick, based on the tick space.

    Liquidity can only be minted or burned at ticks that are multiples of tick_space.
    The active liquidity in the current tick is the liquidity at the greatest multiple
    of tick_space that is <= current_tick. If that tick is not initialized in either profile,
    then there is no active liquidity.

    Returns a tuple (active_passive, active_jit, total_liquidity) as Decimals.
    """
    from decimal import Decimal

    # Compute the active tick: the greatest multiple of tick_space that is <= current_tick.
    active_tick = int((current_tick // tick_space) * tick_space)

    # Get the liquidity at that tick for both passive and JIT providers.
    active_passive = Decimal(passive_liq.get(active_tick, 0))
    active_jit = Decimal(jit_liq.get(active_tick, 0))

    return active_passive, active_jit, active_passive + active_jit


def get_all_ticks(passive_liq, jit_liq):
    """
    Return a sorted list of all ticks from both passive and JIT liquidity profiles.
    """
    return sorted(set(list(passive_liq.keys()) + list(jit_liq.keys())))

def get_rounded_tick(tick: int, tick_space: int):
    """
    Return which minting available tick the specified tick is whithin
    """
    lower_tick = (tick // tick_space) * tick_space
    upper_tick = lower_tick + tick_space
    return lower_tick, upper_tick

def get_next_tick(current_tick: int,
                  tick_space: int,
                  direction: str,
                  dec0: int,
                  dec1:int 
                  ):
    """
    Determine the next tick and corresponding target sqrt(price) in the specified direction.
    For zeroForOne (direction="down"), we need the maximum tick that is < current_tick.
    For oneForZero (direction="up"), we need the minimum tick that is > current_tick.
    If no tick is found, a limit is set (very low for down, very high for up).
    Returns (next_tick, target_sqrtP). next_tick is None if no tick boundary is available.
    """
    lower_tick,_ = get_rounded_tick(current_tick, tick_space)
    if direction == "down":
        # Next boundary going down is the current range's lower tick, so this
        # step uses the current range's liquidity down to that boundary, then
        # the next range's liquidity is loaded below it.
        next_tick = lower_tick
        target_sqrtP = sqrt_price_from_tick(next_tick, dec0, dec1)
    else:  # direction == "up"
        next_tick = lower_tick + tick_space
        target_sqrtP = sqrt_price_from_tick(next_tick, dec0, dec1)
    return next_tick, target_sqrtP
