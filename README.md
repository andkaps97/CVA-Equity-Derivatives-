# CVA Equity Derivatives 

This project implements a complete Monte Carlo framework to price and compute CVA for a small portfolio of equity derivatives (forwards and European puts) on two equity indices (SX5E and AEX). It includes model verification, exposure profiling, CVA aggregation with stepwise hazard rates, and the impact of collateral and initial margin.


## What this script does (m.py)

The script performs four blocks of analysis and produces a set of plots:

1) Model verification
   - Simulates correlated GBM price paths (with dividend yield) for SX5E and AEX
   - Verifies forward valuation against theory
   - Verifies put option valuation against Black–Scholes
   - Verifies empirical correlation vs. target using Fisher CI

2) Exposure and CVA
   - Simulates monthly paths (5y horizon) and computes a netted portfolio value (2 forwards + 2 puts)
   - Builds a discounted monthly EPE profile
   - Computes instrument-level CVA and netted portfolio CVA with stepwise hazard rates

3) Parameter impact
   - Re-computes netted CVA under higher volatility (30%)
   - Re-computes netted CVA under lower correlation (40%)

4) Collateral and initial margin
   - Computes CVA reductions under different collateral posting frequencies
   - Computes CVA reductions under various initial margin levels

The script prints summary tables and shows four plots: EPE profile, collateral frequency vs CVA, initial margin vs CVA, and parameter sensitivity.


## Methods and assumptions

- Price dynamics: Correlated geometric Brownian motion with dividend yield q
  dS/S = (r − q) dt + σ dW, with Corr(dW_SX5E, dW_AEX) = ρ
- Option pricing: Black–Scholes for European puts with dividend yield
- Exposure: Expected positive exposure (EPE) on a monthly grid; portfolio netting applied pathwise
- Counterparty credit: Stepwise-constant hazard rates over [0,1], (1,3], and (3, T]
- CVA: Sum over time of LGD × default_prob(Δt) × discount_factor(t) × EPE(t)
- Collateral: Simple margining rule that sets collateral equal to current expected exposure at discrete frequencies
- Initial margin: Constant IM level reducing exposure at all times




## Repository layout (focus on m.py)

- m.py – all models, simulations, CVA, and plotting live here
- README.md – this file
- requirements.txt – minimal dependencies



## Requirements

- Python 3.9+ (3.11 recommended)
- Packages: numpy, scipy, matplotlib

Install via the provided requirements file.


## Quickstart (Windows, cmd.exe)

1) Create and activate a virtual environment

```
python -m venv .venv
.venv\Scripts\activate
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Run the analysis

```
python m.py
```

You should see console output for Questions 1–4 and four figures appear.


## Configuration knobs (edit inside m.py)

Top-level parameters (search for the main block in `m.py`):
- risk_free_rate: e.g., 0.03
- dividend_yield: e.g., 0.02
- vol_sx5e, vol_aex: base 0.15; sensitivity to 0.30 in Section 3
- correlation: base 0.80; sensitivity to 0.40 in Section 3
- num_simulations: default 100000; reduce if you face memory/CPU constraints
- T: maturity in years, default 5
- hazard_rates: np.array([h0_1y, h1_3y, h3_T]) with LGD = 0.4
- portfolio: strikes, notionals (num_contracts), vols (for puts)

Performance tips
- Lower `num_simulations` (e.g., 20_000) for faster runs; results will be noisier
- Close plot windows to return to the console if the script blocks on figures
- If running on a headless environment, set a non-interactive matplotlib backend at the top of `m.py` (optional):
  
  ```python
  import matplotlib
  matplotlib.use("Agg")  # before importing pyplot
  ```


## Outputs you will see

Console sections with headers:
-  EQUITY MODEL SIMULATION VERIFICATION
  - Forward valuation: theoretical vs simulated (discounted)
  - Put valuation: Black–Scholes vs simulated (discounted)
  - Empirical correlation with 95% CI
-  EXPOSURES & CVA CALCULATIONS
  - Instrument-level CVA
  - Netted portfolio CVA and netting benefit
-  IMPACT OF MODEL PARAMETERS
  - CVA changes under higher vol and lower correlation
-  COLLATERAL IMPACT ON CVA
  - CVA vs collateral frequency
  - CVA vs initial margin levels

Figures (2×2 grid):
- Monthly discounted EPE profile
- CVA vs collateral posting frequency (with baseline line)
- CVA vs initial margin (bars)
- CVA sensitivity (base vs high vol vs low corr)


## Key functions (where to look in m.py)

- black_scholes_put(S, K, T, r, q, sigma): BS price of a European put with dividend yield
- simulate_price_paths(...), simulate_monthly_price_paths(...): GBM simulators with correlation and dividend yield
- calculate_netted_portfolio_value(...): pathwise portfolio value of forwards + puts; correct time-to-maturity handling
- calculate_monthly_epe_profile(...): discounted EPE curve from netted values
- calculate_total_cva(...), calculate_cva_from_exposures(...): CVA from instrument exposures or pre-aggregated EPE
- calculate_total_cva_with_collateral(...), calculate_total_cva_with_initial_margin(...): simple mitigation models
- correlation_confidence_interval(...), simulate_price_paths_and_log_returns(...): correlation diagnostics


## Troubleshooting

- ModuleNotFoundError: Install dependencies with `pip install -r requirements.txt`
- Slow run or MemoryError: Reduce `num_simulations` (e.g., 20_000) or shorten horizon `T`
- Plots don’t show: Ensure you’re not in a headless environment; otherwise switch to a non-interactive backend and save figures instead of showing them


## References

- Black & Scholes (1973). The pricing of options and corporate liabilities
- Standard CVA discretization with piecewise-constant hazard rates (see e.g., Hull/White lecture notes)


## License

Academic/educational use. Replace with your preferred license if needed.

