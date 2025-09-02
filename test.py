from decimal import Decimal
from uniswap_utils.position import Position
from uniswap_utils.swap import Swap
from uniswap_utils.utils import sqrt_price_from_tick, tick_from_sqrt_price


def test_sqrt_price_conversion():
    """Test that tick to sqrt price conversion and back works correctly"""
    print("Testing sqrt price conversion...")

    # Test case: tick = 0, dec0 = 18, dec1 = 18
    tick = 0
    dec0 = 18
    dec1 = 18
    sqrt_price = sqrt_price_from_tick(tick, dec0, dec1)

    # Manual calculation: sqrt_price = (1.0001^(0/2)) / 10^((18-18)/2) = 1.0
    expected_sqrt_price = Decimal("1.0")
    print(f"Tick: {tick}, dec0: {dec0}, dec1: {dec1}")
    print(f"Calculated sqrt_price: {sqrt_price}")
    print(f"Expected sqrt_price: {expected_sqrt_price}")
    print(f"Match: {abs(sqrt_price - expected_sqrt_price) < Decimal('1e-10')}")

    # Test case: tick = 10, dec0 = 6, dec1 = 18
    tick = 10
    dec0 = 6
    dec1 = 18
    sqrt_price = sqrt_price_from_tick(tick, dec0, dec1)

    # Manual calculation:
    # sqrt_price = (1.0001^(10/2)) / 10^((18-6)/2)
    # = (1.0001^5) / 10^(6)
    # = 1.0005 * 1000 = 1000.5
    expected_sqrt_price = Decimal("1.0001") ** Decimal("5") * Decimal("10") ** Decimal(
        "-6"
    )
    print(f"\nTick: {tick}, dec0: {dec0}, dec1: {dec1}")
    print(f"Calculated sqrt_price: {sqrt_price}")
    print(f"Expected sqrt_price: {expected_sqrt_price}")
    print(f"Match: {abs(sqrt_price - expected_sqrt_price) < Decimal('1e-10')}")

    # Test round trip conversion
    recovered_tick = tick_from_sqrt_price(sqrt_price, dec0, dec1)
    print(f"Round trip tick: {recovered_tick}, original: {tick}")
    print(f"Match: {recovered_tick == tick}")
    print()


def test_position_calculation():
    """Test position token calculation"""
    print("Testing position calculation...")

    # Test case: Position in range
    liquidity = Decimal("10")
    dec0 = 18
    dec1 = 18
    lower_tick = tick_from_sqrt_price(0,dec0,dec1)
    upper_tick = tick_from_sqrt_price(2,dec0,dec1)
    current_tick = tick_from_sqrt_price(0.5,dec0,dec1)
    current_sqrtP = sqrt_price_from_tick(current_tick, dec0, dec1)

    position = Position(liquidity, lower_tick , upper_tick)

    token0, token1 = position.tokens(
        current_sqrtP, dec0, dec1
    )

    sqrtP_lower = sqrt_price_from_tick(lower_tick, dec0, dec1)
    sqrtP_upper = sqrt_price_from_tick(upper_tick, dec0, dec1)

    expected_token0 = float(
        liquidity * ((Decimal("1") / current_sqrtP) - (Decimal("1") / sqrtP_upper))
    )
    expected_token1 = float(liquidity * (current_sqrtP - sqrtP_lower))
    print("Alou:", liquidity, current_sqrtP, sqrtP_lower, sqrtP_upper)

    print(
        f"Liquidity: {liquidity}, range: [{lower_tick}, {upper_tick}], current tick: {current_tick}"
    )
    print(f"Calculated token0: {token0}, expected: {expected_token0}")
    print(f"Calculated token1: {token1}, expected: {expected_token1}")
    print(f"Token0 match: {abs(token0 - expected_token0) < 1e-5}")
    print(f"Token1 match: {abs(token1 - expected_token1) < 1e-5}")
    print()


def test_swap_simulation():
    """Test swap simulation""" 
    print("Testing swap simulation...")

    # Test case 1: Simple swap with constant price and negligible price impact
    amount_in = 100
    zeroForOne = True  # token0 in, token1 out
    initial_price = 1.00001
    fee_rate = Decimal("0.003")  # 0.3%
    dec0 = 18
    dec1 = 18
    tick_space = 10
    liquidity = Decimal("10000000")  # 10M liquidity (large enough to minimize price impact)
    position = Position(0, 0 , 10)

    # Create a liquidity profile with a single position at tick 0
    passive_liq = {-10: liquidity, 0: liquidity}

    swap = Swap(
        amount_in,
        zeroForOne,
        initial_price,
        passive_liq,
        tick_space,
        fee_rate,
        dec0,
        dec1
        )

    result = swap.simulate(
        position,
    )

    # Fee calculation
    fee_amount = amount_in * float(fee_rate)  # 100 * 0.003 = 0.3
    amount_after_fee = amount_in - fee_amount  # 100 - 0.3 = 99.7
    
    # With large enough liquidity, output should be very close to input after fees
    # with minimal price impact
    expected_output = amount_after_fee  # 99.7
    expected_passive_fees = fee_amount  # 0.3
    expected_jit_fees = 0.0

    print("Simple swap test (token0 to token1):")
    print(f"Amount in: {amount_in}, zeroForOne: {zeroForOne}, initial price: {initial_price}")
    print(f"Output amount: {result['output_amount']:.6f}, expected: {expected_output}")
    print(f"Passive LP fees: {result['fees_passive_lp']:.6f}, expected: {expected_passive_fees}")
    print(f"JIT LP fees: {result['fees_jit_lp']:.6f}, expected: {expected_jit_fees}")
    print(f"Final sqrt price: {result['final_sqrt_price']:.8f}")
    print(f"Output match: {abs(result['output_amount'] - expected_output) < 0.01}")
    print(f"Fee match: {abs(result['fees_passive_lp'] - expected_passive_fees) < 0.01}")
    print()

    # Test case 2: Crossing a tick boundary
    amount_in = 2000
    zeroForOne = False  # token1 in, token0 out (price increases)
    initial_price = 1.0
    passive_liq = {0: 500_000, 20: 200_000}  # Liquidity at ticks 0 and 20
    tick_space = 20
    fee_rate = Decimal("0.003")
    dec0 = 18
    dec1 = 18
    position2 = Position(500_000, 0, 20)

    swap2 = Swap(
        amount_in,
        zeroForOne,
        initial_price,
        passive_liq,
        tick_space,
        fee_rate,
        dec0,
        dec1
        )

    result = swap2.simulate(
            position2
            )

    # Detailed calculation for a swap that crosses the tick boundary:
    # 1. First segment (tick 0 to 20):
    #    - Total liquidity = 1,000,000
    #    - √P₁ at tick 20 ≈ 1.001000450120021
    #    - Amount of token1 needed to reach tick 20: 1,000,000 * (1.001000450120021 - 1.0) ≈ 1,000.45
    #    - Fee on this amount: 1,000.45 * 0.003 ≈ 3.00135
    #    - Total token1 consumed: 1,000.45 + 3.00135 ≈ 1,003.45
    #    - Token0 output: 1,000,000 * (1/1.0 - 1/1.001000450120021) ≈ 999.55
    #    - Passive LP fee at tick 0: 3.00135 * 0.5 = 1.500675
    #    - JIT LP fee at tick 0: 3.00135 * 0.5 = 1.500675
    # 
    # 2. Second segment (after tick 20):
    #    - Remaining token1: 2,000 - 1,003.45 = 996.55
    #    - Liquidity at tick 20 = 200,000 (only passive)
    #    - Continue with this liquidity...

    # Expected values based on calculations
    expected_output_approx = 1900.0  # Output token0 amount
    expected_passive_fee_approx = 4.8  # Passive LP fees (both ticks)
    expected_jit_fee_approx = 1.5  # JIT LP fees (only at tick 0)
    expected_sqrt_price_approx = 1.006  # Final sqrt price

    print("Complex swap test with both passive and JIT liquidity:")
    print(f"Amount in: {amount_in}, zeroForOne: {zeroForOne}, initial price: {initial_price}")
    print(f"Output amount: {result['output_amount']:.6f}, expected: ~{expected_output_approx}")
    print(f"Passive LP fees: {result['fees_passive_lp']:.6f}, expected: ~{expected_passive_fee_approx}")
    print(f"JIT LP fees: {result['fees_jit_lp']:.6f}, expected: ~{expected_jit_fee_approx}")
    print(f"Final sqrt price: {result['final_sqrt_price']:.6f}, expected: ~{expected_sqrt_price_approx}")
    print(f"Final tick: {result['final_tick']}")
    
    # Verify the results are within expected ranges
    print(f"Output within range: {abs(result['output_amount'] - expected_output_approx) / expected_output_approx < 0.05}")
    print(f"Passive fees within range: {abs(result['fees_passive_lp'] - expected_passive_fee_approx) / expected_passive_fee_approx < 0.1}")
    print(f"JIT fees within range: {abs(result['fees_jit_lp'] - expected_jit_fee_approx) / expected_jit_fee_approx < 0.1}")
    print(f"Final price within range: {abs(result['final_sqrt_price'] - expected_sqrt_price_approx) / expected_sqrt_price_approx < 0.05}")
    print()


if __name__ == "__main__":
    test_sqrt_price_conversion()
    test_position_calculation()
    test_swap_simulation()
