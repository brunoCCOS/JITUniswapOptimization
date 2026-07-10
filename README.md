# JIT Uniswap Optimization

Finds the optimal just-in-time (JIT) liquidity position for a pending Uniswap V3 swap — the tick range and liquidity amount that maximize LP profit given a fixed capital budget.

## What is JIT liquidity?

A JIT LP detects a pending large swap in the mempool, deposits concentrated liquidity into the relevant tick range in the same block, earns fees from the swap, and withdraws immediately. The LP's profit is the fee revenue minus any adverse price impact from holding the position.

This library optimizes the one decision the JIT LP makes: **which tick range to use and how much liquidity to deploy**.

## Architecture

Two optimizers, same interface:

**Analytical** (`optimization/analytical.py`) — closed-form solution derived from mathematical lemmas (ported from MATLAB). Converts the problem to a canonical coordinate frame, enumerates candidate tick ranges between the current price and the no-JIT final price, and applies per-range lemmas that produce an exact optimal liquidity without simulation. Fast.

**Combinatorial** (`optimization/combinatorial.py`) — brute-force: enumerates candidate tick ranges (width-first, inner-first) and for each runs a 1-D ternary search over liquidity using the simulation-based scoring function. Exact for the simulation objective, slower. Caps at `MAX_NUMBER_POS=50` ranges to bound cost.

Both go through `Utility.optimize(method=...)` which scores the final position via simulation before returning.

## Swap simulation

`uniswap_utils/swap.py` replays a Uniswap V3 swap tick-by-tick using **Q96 integer arithmetic** — the same fixed-point representation used on-chain. This handles degenerate pools (e.g. USDC/USDT with 8000+ tick steps) without the precision loss or slowdown of Python `Decimal`. Roughly 56× faster than the original Decimal approach on degenerate pools.

Three caches per `Swap` instance (invalidated on `update_state`):
- `_passive_raw` — passive liquidity dict converted from lib → raw Q96 units once
- `_bsqrt_cache` — tick → raw sqrt price lookup
- `_sim_cache_id` — staleness sentinel

## Key modules

| Module | Role |
|--------|------|
| `uniswap_utils/state.py` | `State` — pool snapshot (price, passive liquidity map, fee rate, decimals) |
| `uniswap_utils/position.py` | `Position` — JIT tick range + liquidity, budget→liquidity conversion |
| `uniswap_utils/swap.py` | `Swap.simulate()` — multi-range swap replay, fee attribution |
| `uniswap_utils/utils.py` | Tick↔sqrt-price arithmetic, active liquidity lookup |
| `optimization/utility.py` | `Utility` — facade that owns `Swap`, dispatches to optimizer, scores result |
| `optimization/analytical.py` | `AnalyticalOptimizer` — closed-form optimizer |
| `optimization/combinatorial.py` | `CombinatorialOptimizer` — simulation-based optimizer |
| `optimization/search.py` | 1-D maximizers (ternary search, golden section, Fibonacci, random) |

## Usage

```python
from uniswap_utils.state import State
from uniswap_utils.swap import Swap
from optimization.utility import Utility

state = State(
    price=...,          # sqrt price in library units
    passive_dict=...,   # {lower_tick: liquidity, ...}
    tick_space=60,
    fee_rate=0.003,
    dec0=6, dec1=18,    # token decimals (e.g. USDC/WETH)
)

swap = Swap(amount_in=1000.0, zeroForOne=True, state=state)
utility = Utility(swap, price0=1.0, price1=2500.0)

result = utility.optimize(method="analytical", budget=10_000.0)
# or method="combinatorial"

print(result)
# {'lower_tick': ..., 'upper_tick': ..., 'liquidity': ..., 'utility': ...}
```

## Setup

```bash
uv sync
python test.py   # or: pytest test.py
```

Requires Python ≥ 3.10 and numpy ≥ 2.2.6.
