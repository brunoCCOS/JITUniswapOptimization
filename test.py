from decimal import Decimal
from uniswap_utils.position import Position
from uniswap_utils.swap import Swap
from uniswap_utils.state import State
from uniswap_utils.utils import sqrt_price_from_tick, tick_from_sqrt_price
from optimization.utility import Utility
from optimization.search import ternary_search_max


def test_sqrt_price_conversion():
    # tick=0, equal decimals → sqrt price = 1
    tick, dec0, dec1 = 0, 18, 18
    sqrt_price = sqrt_price_from_tick(tick, dec0, dec1)
    assert abs(sqrt_price - Decimal("1.0")) < Decimal("1e-10"), f"Expected 1.0, got {sqrt_price}"

    # tick=10, dec0=6, dec1=18
    tick, dec0, dec1 = 10, 6, 18
    sqrt_price = sqrt_price_from_tick(tick, dec0, dec1)
    expected = Decimal("1.0001") ** Decimal("5") * Decimal("10") ** Decimal("-6")
    assert abs(sqrt_price - expected) < Decimal("1e-10"), f"Expected {expected}, got {sqrt_price}"

    # Round-trip: tick → sqrt → tick
    recovered = tick_from_sqrt_price(sqrt_price, dec0, dec1)
    assert recovered == tick, f"Round-trip failed: expected {tick}, got {recovered}"


def test_position_calculation():
    liquidity = Decimal("10")
    dec0, dec1 = 18, 18
    lower_tick, upper_tick, current_tick = -10, 10, 0
    current_sqrtP = sqrt_price_from_tick(current_tick, dec0, dec1)
    sqrtP_lower = sqrt_price_from_tick(lower_tick, dec0, dec1)
    sqrtP_upper = sqrt_price_from_tick(upper_tick, dec0, dec1)

    position = Position(liquidity, lower_tick, upper_tick)
    token0, token1 = position.tokens(current_sqrtP, dec0, dec1)

    expected_token0 = float(liquidity * ((Decimal("1") / current_sqrtP) - (Decimal("1") / sqrtP_upper)))
    expected_token1 = float(liquidity * (current_sqrtP - sqrtP_lower))

    assert abs(token0 - expected_token0) < 1e-5, f"token0: {token0} != {expected_token0}"
    assert abs(token1 - expected_token1) < 1e-5, f"token1: {token1} != {expected_token1}"


def test_swap_simulation_simple():
    """Simple single-range swap: no tick crossing, no JIT. Checks fee accounting.
    Note: output_amount is always 0.0 (not computed — optimizer only needs fees/final price).
    """
    amount_in = 100
    dec0, dec1 = 18, 18
    passive_liq = {-10: Decimal("10000000"), 0: Decimal("10000000")}
    state = State(
        price=1.00001,
        passive_dict=passive_liq,
        tick_space=10,
        fee_rate=Decimal("0.003"),
        dec0=dec0,
        dec1=dec1,
    )
    swap = Swap(amount_in, zeroForOne=True, state=state)
    result = swap.simulate(Position(0, 0, 10))

    expected_fee = amount_in * 0.003  # 0.3
    assert result["output_amount"] == 0.0, "output_amount is not computed (optimizer doesn't need it)"
    assert abs(result["fees_passive_lp"] - expected_fee) < 0.01, \
        f"passive fees {result['fees_passive_lp']:.4f} not near {expected_fee}"
    assert result["fees_jit_lp"] == 0.0, f"expected 0 jit fees, got {result['fees_jit_lp']}"


def test_swap_simulation_crossing():
    """
    Swap that crosses one tick boundary with 50/50 passive+JIT at first range.
    Verifies: boundary crossing, fee splitting, final price.
    """
    amount_in = 2000
    dec0, dec1 = 18, 18
    # passive at tick 0 and tick 20; JIT covers [0, 20)
    passive_liq = {0: 500_000, 20: 1_000_000}
    state = State(
        price=1.0,
        passive_dict=passive_liq,
        tick_space=20,
        fee_rate=Decimal("0.003"),
        dec0=dec0,
        dec1=dec1,
    )
    swap = Swap(amount_in, zeroForOne=False, state=state)
    result = swap.simulate(Position(500_000, 0, 20))

    # At tick [0,20): total L=1e6 (500k passive + 500k JIT).
    # First-range capacity ≈ 1e6 * 0.001 = 1000 net token1, gross ≈ 1003.
    # JIT gets half the first-range fee ≈ 1.5 token1.
    # After tick 20: only 200k passive (position.to_dict gives nothing at tick 20).
    # Total fees_jit should be ≈ 1.5 (only from first range).
    assert result["output_amount"] == 0.0, "output_amount not computed"
    assert abs(result["fees_jit_lp"] - 1.5) < 0.1, \
        f"JIT fees {result['fees_jit_lp']:.4f} not near 1.5"
    assert abs(result["final_sqrt_price"] - 1.002) / 1.002 < 0.05, \
        f"final price {result['final_sqrt_price']:.5f} not near 1.002"
    assert result["fees_passive_lp"] > result["fees_jit_lp"], \
        "passive should earn more fees than JIT (passive also earns post-JIT)"


def test_swap_simulation_down():
    """Down-swap (zeroForOne=True) sanity check."""
    dec0, dec1 = 18, 18
    passive_liq = {-20: 1_000_000, 0: 1_000_000}
    state = State(
        price=1.0,
        passive_dict=passive_liq,
        tick_space=20,
        fee_rate=Decimal("0.003"),
        dec0=dec0,
        dec1=dec1,
    )
    swap = Swap(500.0, zeroForOne=True, state=state)
    result = swap.simulate(Position(0, 0, 20))

    # Price should decrease
    assert result["final_sqrt_price"] < 1.0, \
        f"down-swap should decrease price, got {result['final_sqrt_price']}"
    assert result["fees_passive_lp"] > 0, "should collect passive fees"
    assert result["fees_jit_lp"] == 0.0, "no JIT liquidity, no JIT fees"


def test_swap_thin_pool():
    """
    Thin pool (USDC/USDT style, dec0=dec1=6): verifies Q96 handles ~8000-step
    simulations without stalling and produces a finite result.
    """
    dec0, dec1 = 6, 6
    passive_liq_lib = 8939.0
    # Dense dict covering 2000 tick-ranges above current tick
    passive_dict = {i * 10: passive_liq_lib for i in range(-100, 2000)}
    state = State(
        price=1.0,
        passive_dict=passive_dict,
        tick_space=10,
        fee_rate=500 / 1_000_000,
        dec0=dec0,
        dec1=dec1,
    )
    swap = Swap(amount_in=35000.0, zeroForOne=False, state=state)
    result = swap.simulate(Position(0, 0, 10))

    # Should advance many ticks (thin liquidity → large price move)
    assert result["final_tick"] > 1000, \
        f"expected large tick advance, got final_tick={result['final_tick']}"
    assert result["fees_jit_lp"] == 0.0
    assert result["fees_passive_lp"] > 0


def test_swap_asymmetric_decimals():
    """
    USDC(dec0=6)/WETH(dec1=18) style: verifies unit conversions for dec0 ≠ dec1.
    sqrt_adj = 10^6, liq_scale = 10^12.

    At tick=0, lib_sqrt = 1e-6.  passive_liq_lib = 1e6 → raw_L = 1e18.
    amount_in = 0.001 WETH lib → raw = 1e15 token1.  Caps ~5e14/step → ~2 steps.
    """
    dec0, dec1 = 6, 18
    passive_liq = {i * 10: 1_000_000.0 for i in range(-10, 20)}  # dense around tick 0
    state = State(
        price=1e-6,      # lib_sqrt at tick 0: sqrt(1.0001^0) * 10^((6-18)/2) = 1e-6
        passive_dict=passive_liq,
        tick_space=10,
        fee_rate=500 / 1_000_000,
        dec0=dec0,
        dec1=dec1,
    )
    swap = Swap(amount_in=0.001, zeroForOne=False, state=state)
    result = swap.simulate(Position(0, 0, 10))

    # lib_sqrt at tick 0 is 1e-6; a small up-swap should give a slightly larger value
    assert result["final_sqrt_price"] > 1e-6, \
        f"price should increase for up-swap, got {result['final_sqrt_price']}"
    assert result["final_sqrt_price"] < 2e-6, \
        f"price moved too far: {result['final_sqrt_price']}"
    assert result["fees_passive_lp"] > 0, "should earn passive fees"
    # fees should be in WETH lib units (~0.001 * 0.0005 = 5e-7)
    assert result["fees_passive_lp"] < 0.001, \
        f"fees suspiciously large: {result['fees_passive_lp']}"


def _decimal_simulate(swap, position):
    """
    Reference implementation using the original Decimal arithmetic.
    Mirrors the old simulate() logic exactly so we can regression-test the Q96 rewrite.
    """
    from decimal import Decimal, getcontext
    from uniswap_utils.utils import (
        sqrt_price_from_tick, tick_from_sqrt_price,
        get_next_tick, calculate_active_liquidity,
    )
    getcontext().prec = 28

    state = swap.state
    remaining = Decimal(swap.amount_in)
    current_sqrt = Decimal(state.price)
    current_tick = tick_from_sqrt_price(current_sqrt, state.dec0, state.dec1)
    jit_liq = position.to_dict(state.tick_space)
    fee_rate = Decimal(state.fee_rate)
    fees_passive = Decimal(0)
    fees_jit = Decimal(0)

    while remaining > 0:
        direction = "down" if swap.zeroForOne else "up"
        boundary_tick, target_sqrt = get_next_tick(
            current_tick, state.tick_space, direction, state.dec0, state.dec1
        )
        liq_P, liq_J, L = calculate_active_liquidity(
            current_tick, state.passive_dict, jit_liq, state.tick_space
        )
        if L == 0:
            break
        gross_in = remaining / (1 + fee_rate)
        if not swap.zeroForOne:
            available = L * (target_sqrt - current_sqrt)
            if available > gross_in:
                actual = gross_in
                fee = remaining - gross_in
                current_sqrt = current_sqrt + actual / L
                current_tick = tick_from_sqrt_price(current_sqrt, state.dec0, state.dec1)
                remaining = Decimal(0)
            else:
                actual = available
                fee = actual * fee_rate
                remaining -= actual * (1 + fee_rate)
                current_sqrt = target_sqrt
                current_tick = boundary_tick
        else:
            available = L * (1 / target_sqrt - 1 / current_sqrt)
            if available > gross_in:
                actual = gross_in
                fee = remaining - gross_in
                inv_new = 1 / current_sqrt + actual / L
                current_sqrt = 1 / inv_new
                current_tick = tick_from_sqrt_price(current_sqrt, state.dec0, state.dec1)
                remaining = Decimal(0)
            else:
                actual = available
                fee = actual * fee_rate
                remaining -= actual * (1 + fee_rate)
                current_sqrt = target_sqrt
                current_tick = boundary_tick - 2
        fees_passive += liq_P / L * fee
        fees_jit += liq_J / L * fee

    return {
        "final_sqrt_price": float(current_sqrt),
        "final_tick": current_tick,
        "fees_passive_lp": float(fees_passive),
        "fees_jit_lp": float(fees_jit),
    }


def test_q96_vs_decimal_regression():
    """
    Q96 simulate() must agree with the original Decimal reference on key outputs:
    final_sqrt_price and fees_jit_lp within 0.05% relative tolerance.
    Covers up-swap, down-swap, and equal/asymmetric decimals.
    """
    cases = [
        # (desc, amount_in, zeroForOne, price, passive_dict, position, dec0, dec1, ts, fee_rate)
        (
            "up-swap equal dec, 1 range",
            500.0, False, 1.0,
            {0: 1_000_000.0, 20: 500_000.0},
            Position(200_000, 0, 20),
            18, 18, 20, 0.003,
        ),
        (
            "down-swap equal dec, crossing",
            800.0, True, 1.0,
            {-20: 800_000.0, 0: 1_200_000.0},
            Position(300_000, -20, 0),
            18, 18, 20, 0.003,
        ),
        (
            "up-swap no JIT multi-range",
            2000.0, False, 1.0,
            {0: 700_000.0, 20: 1_500_000.0, 40: 700_000.0},
            Position(0, 0, 20),
            18, 18, 20, 0.003,
        ),
        (
            "asymmetric dec (USDC/WETH style), terminates in range",
            0.5e-6, False, 1e-6,  # price in lib units, amount in lib
            {i * 10: 1_000.0 for i in range(-5, 20)},
            Position(200.0, 0, 10),
            6, 18, 10, 0.003,
        ),
    ]

    for desc, amount_in, zeroForOne, price, passive_dict, position, dec0, dec1, ts, fee_rate in cases:
        state = State(
            price=price, passive_dict=passive_dict, tick_space=ts,
            fee_rate=Decimal(str(fee_rate)), dec0=dec0, dec1=dec1,
        )
        swap = Swap(amount_in, zeroForOne, state)

        ref = _decimal_simulate(swap, position)
        got = swap.simulate(position)

        rtol = 5e-4  # 0.05% tolerance

        assert ref["final_sqrt_price"] == 0 or abs(got["final_sqrt_price"] - ref["final_sqrt_price"]) / abs(ref["final_sqrt_price"]) < rtol, \
            f"[{desc}] final_sqrt_price: Q96={got['final_sqrt_price']:.8g} ref={ref['final_sqrt_price']:.8g}"

        if ref["fees_jit_lp"] > 1e-15:
            assert abs(got["fees_jit_lp"] - ref["fees_jit_lp"]) / ref["fees_jit_lp"] < rtol, \
                f"[{desc}] fees_jit_lp: Q96={got['fees_jit_lp']:.8g} ref={ref['fees_jit_lp']:.8g}"

        if ref["fees_passive_lp"] > 1e-15:
            assert abs(got["fees_passive_lp"] - ref["fees_passive_lp"]) / ref["fees_passive_lp"] < rtol, \
                f"[{desc}] fees_passive_lp: Q96={got['fees_passive_lp']:.8g} ref={ref['fees_passive_lp']:.8g}"


def test_optimization():
    """Combinatorial optimizer returns a valid position."""
    dec0, dec1 = 18, 18
    passive_liq = {0: 700_000.0, 20: 1_500_000.0, 40: 700_000}
    state = State(
        price=1.0,
        passive_dict=passive_liq,
        tick_space=20,
        fee_rate=Decimal("0.003"),
        dec0=dec0,
        dec1=dec1,
    )
    swap = Swap(2000, zeroForOne=False, state=state)
    utility = Utility(swap, price0=1.0, price1=1.0)
    result = utility.optimize(1000.0, method="combinatorial", opt_func=ternary_search_max, epsilon=1e-6)

    assert result.get("lower_tick") is not None, "optimizer returned no position"
    assert result["utility"] > 0, f"expected positive utility, got {result['utility']}"


if __name__ == "__main__":
    test_sqrt_price_conversion()
    print("test_sqrt_price_conversion PASSED")

    test_position_calculation()
    print("test_position_calculation PASSED")

    test_swap_simulation_simple()
    print("test_swap_simulation_simple PASSED")

    test_swap_simulation_crossing()
    print("test_swap_simulation_crossing PASSED")

    test_swap_simulation_down()
    print("test_swap_simulation_down PASSED")

    test_swap_thin_pool()
    print("test_swap_thin_pool PASSED")

    test_swap_asymmetric_decimals()
    print("test_swap_asymmetric_decimals PASSED")

    test_q96_vs_decimal_regression()
    print("test_q96_vs_decimal_regression PASSED")

    test_optimization()
    print("test_optimization PASSED")

    print("\nAll tests passed.")
