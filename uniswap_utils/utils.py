from decimal import Decimal
import os


def print_debug(msg):
    if os.getenv("DEBUG") == "1":
        print(msg)

def sqrt_price_from_tick(tick, dec0, dec1):
    """
    Compute actual sqrt(price) from a given tick index, accounting for token decimals.
        sqrt(P) = 1.0001^(tick/2) / 10^((dec1 - dec0)/2)
    where P = price of token1 in terms of token0.
    """
    sqrt_price = (Decimal("1.0001") ** (Decimal(tick) / 2)) / (
        Decimal(10) ** ((Decimal(dec1) - Decimal(dec0)) / 2)
    )
    return sqrt_price


def tick_from_sqrt_price(sqrt_price, dec0, dec1):
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


def calculate_active_liquidity(current_tick, passive_liq, jit_liq, tick_space):
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


def get_next_tick(current_tick, tick_list, direction, dec0, dec1):
    """
    Determine the next tick and corresponding target sqrt(price) in the specified direction.
    For zeroForOne (direction="down"), we need the maximum tick that is < current_tick.
    For oneForZero (direction="up"), we need the minimum tick that is > current_tick.
    If no tick is found, a limit is set (very low for down, very high for up).
    Returns (next_tick, target_sqrtP). next_tick is None if no tick boundary is available.
    """
    if direction == "down":
        lower_ticks = [t for t in tick_list if t < current_tick]
        if lower_ticks:
            next_tick = max(lower_ticks)
            target_sqrtP = sqrt_price_from_tick(next_tick, dec0, dec1)
        else:
            next_tick = None
            target_sqrtP = Decimal("1e-20")  # Effectively zero
    else:  # direction == "up"
        upper_ticks = [t for t in tick_list if t > current_tick]
        if upper_ticks:
            next_tick = min(upper_ticks)
            target_sqrtP = sqrt_price_from_tick(next_tick, dec0, dec1)
        else:
            next_tick = None
            target_sqrtP = Decimal("1e20")  # Effectively infinite
    return next_tick, target_sqrtP
