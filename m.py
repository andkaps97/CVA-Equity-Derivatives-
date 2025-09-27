"""
CVA Equity Derivatives
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# Function to calculate the Black-Scholes price for a put option
def black_scholes_put(S, K, T, r, q, sigma, epsilon=1e-10):
    """Calculate Black-Scholes put option price"""
    T = max(T, epsilon)
    sigma = max(sigma, epsilon)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price


# Function to simulate equity price paths (annual steps)
def simulate_price_paths(S0, T, r, q, sigma, rho, num_simulations, dt, path):
    """Simulate annual price paths for equity indices"""
    price_paths = np.zeros((num_simulations, T + 1))
    price_paths[:, 0] = S0

    # Generate correlated random variables
    cholesky_matrix = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))
    for t in range(1, T + 1):
        z = np.dot(cholesky_matrix, np.random.normal(size=(2, num_simulations)))
        z_sx5e, z_aex = z[0], z[1]
        if path == 'SX5E':
            price_paths[:, t] = price_paths[:, t - 1] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * z_sx5e * np.sqrt(dt))
        else:
            price_paths[:, t] = price_paths[:, t - 1] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * z_aex * np.sqrt(dt))
    return price_paths


# FIXED: Function to simulate monthly price paths with dividend yield
def simulate_monthly_price_paths(S0, T, r, q, sigma, rho, num_simulations, path, num_steps=12):
    """Simulate monthly price paths with proper dividend yield inclusion"""
    dt = 1 / num_steps  # Monthly steps in terms of years
    total_steps = int(T * num_steps)
    price_paths = np.zeros((num_simulations, total_steps + 1))
    price_paths[:, 0] = S0

    cholesky_matrix = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))

    for t in range(1, total_steps + 1):
        z = np.dot(cholesky_matrix, np.random.normal(size=(2, num_simulations)))
        z_sx5e, z_aex = z[0], z[1]

        if path == 'SX5E':
            # Include dividend yield q in the drift
            price_paths[:, t] = price_paths[:, t - 1] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * z_sx5e * np.sqrt(dt)
            )
        else:  # AEX
            price_paths[:, t] = price_paths[:, t - 1] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * z_aex * np.sqrt(dt)
            )

    return price_paths


# Function to calculate discounted payoff for forwards
def calculate_discounted_payoff_forwards(price_paths, S0, K, r, T, num_contracts):
    """Calculate discounted expected payoff for forward contracts"""
    expected_payoff = np.mean(price_paths[:, -1]) - K
    discounted_payoff = expected_payoff * np.exp(-r * T)
    return discounted_payoff


# Function to calculate discounted payoff for put options
def calculate_discounted_payoff_puts(price_paths, K, r, T, num_contracts):
    """Calculate discounted expected payoff for put options"""
    payoff_put = np.maximum(K - price_paths[:, -1], 0)
    expected_payoff_put = np.mean(payoff_put)
    discounted_payoff_put = expected_payoff_put * np.exp(-r * T)
    return discounted_payoff_put, payoff_put


# FIXED: Function to calculate netted portfolio value with correct time-to-maturity
def calculate_netted_portfolio_value(price_paths_sx5e, price_paths_aex, T, num_simulations, portfolio, steps):
    """Calculate netted portfolio value with proper time-to-maturity calculation"""
    num_steps = T * steps
    netted_portfolio_values = np.zeros((num_simulations, num_steps + 1))

    for t in range(1, num_steps + 1):
        # Forwards are simple: S_t - K
        forward_sx5e_value = (price_paths_sx5e[:, t] - portfolio['SX5E_forward']['K']) * portfolio['SX5E_forward'][
            'num_contracts']
        forward_aex_value = (price_paths_aex[:, t] - portfolio['AEX_forward']['K']) * portfolio['AEX_forward'][
            'num_contracts']

        # FIXED: Corrected time-to-maturity calculation (t is in months, so divide by 12)
        time_remaining = max(T - t / 12, 0)  # Time remaining in years

        if time_remaining > 0:
            put_sx5e_value = black_scholes_put(
                price_paths_sx5e[:, t],
                portfolio['SX5E_put']['K'],
                time_remaining,  # Corrected time
                risk_free_rate,
                dividend_yield,
                portfolio['SX5E_put']['vol']
            ) * portfolio['SX5E_put']['num_contracts']

            put_aex_value = black_scholes_put(
                price_paths_aex[:, t],
                portfolio['AEX_put']['K'],
                time_remaining,  # Corrected time
                risk_free_rate,
                dividend_yield,
                portfolio['AEX_put']['vol']
            ) * portfolio['AEX_put']['num_contracts']
        else:
            # At maturity, use intrinsic value
            put_sx5e_value = np.maximum(portfolio['SX5E_put']['K'] - price_paths_sx5e[:, t], 0) * portfolio['SX5E_put'][
                'num_contracts']
            put_aex_value = np.maximum(portfolio['AEX_put']['K'] - price_paths_aex[:, t], 0) * portfolio['AEX_put'][
                'num_contracts']

        netted_value = forward_sx5e_value + forward_aex_value + put_sx5e_value + put_aex_value
        netted_portfolio_values[:, t] = netted_value

    return netted_portfolio_values


# Function to calculate the monthly expected positive exposure profile
def calculate_monthly_epe_profile(netted_portfolio_values, T, r, steps):
    """Calculate discounted EPE profile"""
    num_steps = T * steps
    positive_exposures = np.maximum(netted_portfolio_values, 0)
    monthly_epe_profile = positive_exposures.mean(axis=0)
    # Discount factors for each month
    discount_factors = np.exp(-r * np.arange(num_steps + 1) / steps)
    discounted_epe_profile = monthly_epe_profile * discount_factors
    return discounted_epe_profile


# Function to calculate the default probability
def default_probability(hazard_rate, dt):
    """Calculate default probability for a time step"""
    return 1 - np.exp(-hazard_rate * dt)


# Function to calculate CVA charge for a single time step
def cva_charge_single_step(expected_exposure, default_prob, lgd, discount_factor):
    """Calculate CVA contribution for a single time step"""
    return lgd * default_prob * discount_factor * expected_exposure


# Function to get the appropriate hazard rate for a given time
def get_hazard_rate(time, hazard_rate_intervals):
    """Get hazard rate based on time intervals"""
    if time <= 1:
        return hazard_rate_intervals[0]
    elif time <= 3:
        return hazard_rate_intervals[1]
    else:
        return hazard_rate_intervals[2]


# FIXED: Separate function for portfolio CVA calculation
def calculate_cva_from_exposures(exposures, hazard_rate_intervals, lgd, r, T, dt):
    """Calculate CVA from pre-computed exposures (for netted portfolio)"""
    cva_total = 0
    num_steps = len(exposures) - 1

    for step in range(1, num_steps + 1):
        time = step * dt
        hazard_rate = get_hazard_rate(time, hazard_rate_intervals)

        # Use pre-computed exposure (already expected value)
        expected_exposure = exposures[step]

        default_prob = default_probability(hazard_rate, dt)
        discount_factor = np.exp(-r * time)
        cva_charge = cva_charge_single_step(expected_exposure, default_prob, lgd, discount_factor)
        cva_total += cva_charge

    return cva_total


# Function to calculate total CVA for individual instruments
def calculate_total_cva(price_paths, hazard_rate_intervals, lgd, r, T, dt, instrument=None):
    """Calculate total CVA charge for individual instruments"""
    cva_total = 0
    num_steps = int(T / dt)

    for step in range(1, num_steps + 1):
        time = step * dt
        hazard_rate = get_hazard_rate(time, hazard_rate_intervals)

        if instrument and 'type' in instrument:
            if instrument['type'] == 'put':
                time_remaining = max(T - time, 0)
                if time_remaining > 0:
                    put_prices = black_scholes_put(
                        price_paths[:, step],
                        instrument['K'],
                        time_remaining,
                        r,
                        instrument.get('q', 0),
                        instrument['vol']
                    )
                else:
                    put_prices = np.maximum(instrument['K'] - price_paths[:, step], 0)
                expected_positive_exposure = put_prices * instrument['num_contracts']
            else:  # Forward
                expected_positive_exposure = np.maximum(price_paths[:, step] - instrument['K'], 0) * instrument[
                    'num_contracts']

            expected_positive_exposure_mean = np.mean(np.maximum(expected_positive_exposure, 0))
        else:
            # For portfolio values passed directly
            expected_positive_exposure_mean = np.mean(np.maximum(price_paths[:, step], 0))

        default_prob = default_probability(hazard_rate, dt)
        discount_factor = np.exp(-r * time)
        cva_charge = cva_charge_single_step(expected_positive_exposure_mean, default_prob, lgd, discount_factor)
        cva_total += cva_charge

    return cva_total


# Confidence interval functions
def confidence_interval(data, confidence=0.95):
    """Calculate confidence interval for data"""
    z_score = norm.ppf((1 + confidence) / 2)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    margin_error = z_score * (sample_std / np.sqrt(len(data)))
    return sample_mean - margin_error, sample_mean + margin_error


def correlation_confidence_interval(r, n, confidence=0.95):
    """Calculate confidence interval for correlation using Fisher's z-transformation"""
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_score = norm.ppf((1 + confidence) / 2)
    delta = z_score * se
    lower = np.tanh(z - delta)
    upper = np.tanh(z + delta)
    return lower, upper


# Function to simulate and get log returns for correlation verification
def simulate_price_paths_and_log_returns(S0_sx5e, S0_aex, T, r, q, sigma_sx5e, sigma_aex, rho, num_simulations, dt):
    """Simulate price paths and calculate log returns"""
    price_paths_sx5e = np.zeros((num_simulations, T + 1))
    price_paths_aex = np.zeros((num_simulations, T + 1))
    price_paths_sx5e[:, 0] = S0_sx5e
    price_paths_aex[:, 0] = S0_aex

    log_returns_sx5e = np.zeros((num_simulations, T))
    log_returns_aex = np.zeros((num_simulations, T))

    cholesky_matrix = np.linalg.cholesky(np.array([[1, rho], [rho, 1]]))

    for t in range(1, T + 1):
        z = np.dot(cholesky_matrix, np.random.normal(size=(2, num_simulations)))
        z_sx5e, z_aex = z[0], z[1]

        # SX5E
        price_paths_sx5e[:, t] = price_paths_sx5e[:, t - 1] * np.exp(
            (r - q - 0.5 * sigma_sx5e ** 2) * dt + sigma_sx5e * z_sx5e * np.sqrt(dt)
        )
        log_returns_sx5e[:, t - 1] = np.log(price_paths_sx5e[:, t] / price_paths_sx5e[:, t - 1])

        # AEX
        price_paths_aex[:, t] = price_paths_aex[:, t - 1] * np.exp(
            (r - q - 0.5 * sigma_aex ** 2) * dt + sigma_aex * z_aex * np.sqrt(dt)
        )
        log_returns_aex[:, t - 1] = np.log(price_paths_aex[:, t] / price_paths_aex[:, t - 1])

    return log_returns_sx5e, log_returns_aex


# Collateral and margin functions remain the same
def calculate_total_cva_with_collateral(exposures, hazard_rate_intervals, lgd, r, T, dt, collateral_frequency):
    """Calculate CVA with collateral posting"""
    cva_total = 0
    num_steps = len(exposures) - 1
    collateral_posted = 0

    for step in range(1, num_steps + 1):
        time = step * dt
        hazard_rate = get_hazard_rate(time, hazard_rate_intervals)

        expected_exposure = exposures[step]

        # Update collateral at specified frequency
        if step % collateral_frequency == 0 or step == 1:
            collateral_posted = expected_exposure

        mitigated_exposure = max(expected_exposure - collateral_posted, 0)

        default_prob = default_probability(hazard_rate, dt)
        discount_factor = np.exp(-r * time)
        cva_charge = cva_charge_single_step(mitigated_exposure, default_prob, lgd, discount_factor)
        cva_total += cva_charge

    return cva_total


def calculate_total_cva_with_initial_margin(exposures, hazard_rate_intervals, lgd, r, T, dt, initial_margin):
    """Calculate CVA with initial margin"""
    cva_total = 0
    num_steps = len(exposures) - 1

    for step in range(1, num_steps + 1):
        time = step * dt
        hazard_rate = get_hazard_rate(time, hazard_rate_intervals)

        expected_exposure = exposures[step]
        mitigated_exposure = max(expected_exposure - initial_margin, 0)

        default_prob = default_probability(hazard_rate, dt)
        discount_factor = np.exp(-r * time)
        cva_charge = cva_charge_single_step(mitigated_exposure, default_prob, lgd, discount_factor)
        cva_total += cva_charge

    return cva_total


def increment_hazard_rates(hazard_rates, increment, interval):
    """Increment hazard rate for a specific interval"""
    new_hazard_rates = hazard_rates.copy()
    new_hazard_rates[interval] += increment
    return new_hazard_rates


# Main execution
if __name__ == "__main__":
    # Given parameters
    risk_free_rate = 0.03
    dt = 1 / 12  # Monthly time step
    dividend_yield = 0.02
    vol_sx5e = 0.15
    vol_aex = 0.15
    correlation = 0.80
    num_simulations = 100000
    T = 5  # Time to maturity in years

    # Forward hazard rates
    hazard_rates = np.array([0.02, 0.0215, 0.0220])
    lgd = 0.4

    np.random.seed(70)  # For reproducibility

    # Portfolio specifications
    portfolio = {
        'SX5E_forward': {'S0': 4235, 'K': 4235, 'num_contracts': 10000, 'q': dividend_yield, 'type': 'forward'},
        'AEX_forward': {'S0': 770, 'K': 770, 'num_contracts': 55000, 'q': dividend_yield, 'type': 'forward'},
        'SX5E_put': {'S0': 4235, 'K': 3388, 'num_contracts': 10000, 'vol': vol_sx5e, 'q': dividend_yield,
                     'type': 'put'},
        'AEX_put': {'S0': 770, 'K': 616, 'num_contracts': 55000, 'vol': vol_aex, 'q': dividend_yield, 'type': 'put'}
    }

    print("=" * 80)
    print("CVA EQUITY DERIVATIVES ")
    print("=" * 80)

    # Question 1: Model Verification
    print("\n" + "=" * 60)
    print("EQUITY MODEL SIMULATION VERIFICATION")
    print("=" * 60)

    # Annual simulations for verification
    price_paths_sx5e = simulate_price_paths(portfolio['SX5E_forward']['S0'], T, risk_free_rate,
                                            dividend_yield, vol_sx5e, correlation, num_simulations, 1, 'SX5E')
    price_paths_aex = simulate_price_paths(portfolio['AEX_forward']['S0'], T, risk_free_rate,
                                           dividend_yield, vol_aex, correlation, num_simulations, 1, 'AEX')

    # 1a) Forward valuation verification
    theoretical_sx5e = portfolio['SX5E_forward']['S0'] * np.exp((risk_free_rate - dividend_yield) * T)
    theoretical_aex = portfolio['AEX_forward']['S0'] * np.exp((risk_free_rate - dividend_yield) * T)

    theoretical_valuation_sx5e = (theoretical_sx5e - portfolio['SX5E_forward']['K']) * np.exp(-risk_free_rate * T)
    theoretical_valuation_aex = (theoretical_aex - portfolio['AEX_forward']['K']) * np.exp(-risk_free_rate * T)

    discounted_payoff_sx5e_forward = calculate_discounted_payoff_forwards(
        price_paths_sx5e, portfolio['SX5E_forward']['S0'], portfolio['SX5E_forward']['K'],
        risk_free_rate, T, portfolio['SX5E_forward']['num_contracts']
    )
    discounted_payoff_aex_forward = calculate_discounted_payoff_forwards(
        price_paths_aex, portfolio['AEX_forward']['S0'], portfolio['AEX_forward']['K'],
        risk_free_rate, T, portfolio['AEX_forward']['num_contracts']
    )

    print("\n1a) Forward Contracts Verification:")
    print(
        f"SX5E Forward - Theoretical: {theoretical_valuation_sx5e:.4f}, Simulated: {discounted_payoff_sx5e_forward:.4f}")
    print(f"AEX Forward - Theoretical: {theoretical_valuation_aex:.4f}, Simulated: {discounted_payoff_aex_forward:.4f}")

    # 1b) Put option valuation verification
    discounted_payoff_sx5e_put, _ = calculate_discounted_payoff_puts(
        price_paths_sx5e, portfolio['SX5E_put']['K'], risk_free_rate, T, portfolio['SX5E_put']['num_contracts']
    )
    discounted_payoff_aex_put, _ = calculate_discounted_payoff_puts(
        price_paths_aex, portfolio['AEX_put']['K'], risk_free_rate, T, portfolio['AEX_put']['num_contracts']
    )

    bs_price_put_sx5e = black_scholes_put(portfolio['SX5E_put']['S0'], portfolio['SX5E_put']['K'],
                                          T, risk_free_rate, dividend_yield, portfolio['SX5E_put']['vol'])
    bs_price_put_aex = black_scholes_put(portfolio['AEX_put']['S0'], portfolio['AEX_put']['K'],
                                         T, risk_free_rate, dividend_yield, portfolio['AEX_put']['vol'])

    print("\n1b) Put Options Verification:")
    print(f"SX5E Put - Black-Scholes: {bs_price_put_sx5e:.4f}, Simulated: {discounted_payoff_sx5e_put:.4f}")
    print(f"AEX Put - Black-Scholes: {bs_price_put_aex:.4f}, Simulated: {discounted_payoff_aex_put:.4f}")

    # 1c) Correlation verification
    log_returns_sx5e, log_returns_aex = simulate_price_paths_and_log_returns(
        portfolio['SX5E_forward']['S0'], portfolio['AEX_forward']['S0'], T, risk_free_rate, dividend_yield,
        vol_sx5e, vol_aex, correlation, num_simulations, 1
    )

    empirical_correlation = np.corrcoef(log_returns_sx5e.ravel(), log_returns_aex.ravel())[0, 1]
    ci_correlation = correlation_confidence_interval(empirical_correlation, num_simulations)

    print("\n1c) Correlation Verification:")
    print(f"Target Correlation: {correlation:.4f}")
    print(f"Empirical Correlation: {empirical_correlation:.4f}")
    print(f"95% CI for Correlation: [{ci_correlation[0]:.4f}, {ci_correlation[1]:.4f}]")

    #  EPE Profile & CVA Calculations
    print("\n" + "=" * 60)
    print(" EXPOSURES & CVA CALCULATIONS")
    print("=" * 60)

    # Monthly simulations for CVA (with corrected dividend yield)
    monthly_paths_sx5e = simulate_monthly_price_paths(portfolio['SX5E_forward']['S0'], T, risk_free_rate,
                                                      dividend_yield, vol_sx5e, correlation, num_simulations, 'SX5E')
    monthly_paths_aex = simulate_monthly_price_paths(portfolio['AEX_forward']['S0'], T, risk_free_rate,
                                                     dividend_yield, vol_aex, correlation, num_simulations, 'AEX')

    # 2a) Netted portfolio EPE profile
    netted_portfolio_values = calculate_netted_portfolio_value(monthly_paths_sx5e, monthly_paths_aex,
                                                               T, num_simulations, portfolio, 12)
    discounted_epe_profile = calculate_monthly_epe_profile(netted_portfolio_values, T, risk_free_rate, 12)

    # 2b) Individual CVA charges
    cva_sx5e_forward = calculate_total_cva(monthly_paths_sx5e, hazard_rates, lgd, risk_free_rate,
                                           T, dt, portfolio['SX5E_forward'])
    cva_aex_forward = calculate_total_cva(monthly_paths_aex, hazard_rates, lgd, risk_free_rate,
                                          T, dt, portfolio['AEX_forward'])
    cva_sx5e_put = calculate_total_cva(monthly_paths_sx5e, hazard_rates, lgd, risk_free_rate,
                                       T, dt, portfolio['SX5E_put'])
    cva_aex_put = calculate_total_cva(monthly_paths_aex, hazard_rates, lgd, risk_free_rate,
                                      T, dt, portfolio['AEX_put'])

    # 2c) Unnetted portfolio CVA
    cva_unnetted_portfolio = cva_sx5e_forward + cva_aex_forward + cva_sx5e_put + cva_aex_put

    # 2d) Netted portfolio CVA (using corrected function)
    epe_profile = np.mean(np.maximum(netted_portfolio_values, 0), axis=0)
    cva_netted_portfolio = calculate_cva_from_exposures(epe_profile, hazard_rates, lgd, risk_free_rate, T, dt)

    print("\n2b) Individual CVA Charges:")
    print(f"SX5E Forward: {cva_sx5e_forward:.2f}")
    print(f"AEX Forward: {cva_aex_forward:.2f}")
    print(f"SX5E Put: {cva_sx5e_put:.2f}")
    print(f"AEX Put: {cva_aex_put:.2f}")

    print("\n2c) Unnetted Portfolio CVA: {:.2f}".format(cva_unnetted_portfolio))
    print("2d) Netted Portfolio CVA: {:.2f}".format(cva_netted_portfolio))
    print("Netting Benefit: {:.2f} ({:.1f}% reduction)".format(
        cva_unnetted_portfolio - cva_netted_portfolio,
        100 * (cva_unnetted_portfolio - cva_netted_portfolio) / cva_unnetted_portfolio
    ))

    #  Impact of Model Parameters
    print("\n" + "=" * 60)
    print("QUESTION 3: IMPACT OF MODEL PARAMETERS")
    print("=" * 60)

    # 3a) Increased volatility (30%)
    vol_sx5e_new = 0.30
    vol_aex_new = 0.30

    portfolio_3a = portfolio.copy()
    portfolio_3a['SX5E_put']['vol'] = 0.30
    portfolio_3a['AEX_put']['vol'] = 0.30

    monthly_paths_sx5e_3a = simulate_monthly_price_paths(portfolio['SX5E_forward']['S0'], T, risk_free_rate,
                                                         dividend_yield, vol_sx5e_new, correlation, num_simulations,
                                                         'SX5E')
    monthly_paths_aex_3a = simulate_monthly_price_paths(portfolio['AEX_forward']['S0'], T, risk_free_rate,
                                                        dividend_yield, vol_aex_new, correlation, num_simulations,
                                                        'AEX')

    netted_portfolio_values_3a = calculate_netted_portfolio_value(monthly_paths_sx5e_3a, monthly_paths_aex_3a,
                                                                  T, num_simulations, portfolio_3a, 12)
    epe_profile_3a = np.mean(np.maximum(netted_portfolio_values_3a, 0), axis=0)
    cva_netted_3a = calculate_cva_from_exposures(epe_profile_3a, hazard_rates, lgd, risk_free_rate, T, dt)

    # 3b) Reduced correlation (40%)
    corr_new = 0.40

    monthly_paths_sx5e_3b = simulate_monthly_price_paths(portfolio['SX5E_forward']['S0'], T, risk_free_rate,
                                                         dividend_yield, vol_sx5e, corr_new, num_simulations, 'SX5E')
    monthly_paths_aex_3b = simulate_monthly_price_paths(portfolio['AEX_forward']['S0'], T, risk_free_rate,
                                                        dividend_yield, vol_aex, corr_new, num_simulations, 'AEX')

    netted_portfolio_values_3b = calculate_netted_portfolio_value(monthly_paths_sx5e_3b, monthly_paths_aex_3b,
                                                                  T, num_simulations, portfolio, 12)
    epe_profile_3b = np.mean(np.maximum(netted_portfolio_values_3b, 0), axis=0)
    cva_netted_3b = calculate_cva_from_exposures(epe_profile_3b, hazard_rates, lgd, risk_free_rate, T, dt)

    print("\n3a) Volatility increased to 30%:")
    print(f"Original CVA (15% vol): {cva_netted_portfolio:.2f}")
    print(f"New CVA (30% vol): {cva_netted_3a:.2f}")
    print(
        f"Change: {cva_netted_3a - cva_netted_portfolio:.2f} ({100 * (cva_netted_3a / cva_netted_portfolio - 1):.1f}% increase)")

    print("\n3b) Correlation reduced to 40%:")
    print(f"Original CVA (80% corr): {cva_netted_portfolio:.2f}")
    print(f"New CVA (40% corr): {cva_netted_3b:.2f}")
    print(
        f"Change: {cva_netted_3b - cva_netted_portfolio:.2f} ({100 * (cva_netted_3b / cva_netted_portfolio - 1):.1f}% increase)")

    print("\n3c) Commentary:")
    print("- Higher volatility increases CVA due to larger potential exposures")
    print("- Lower correlation increases CVA due to reduced netting benefits")
    print("- Both effects demonstrate the importance of accurate parameter estimation")

    #  Collateral Impact
    print("\n" + "=" * 60)
    print("COLLATERAL IMPACT ON CVA")
    print("=" * 60)

    collateral_frequencies = [1, 2, 3, 4, 6, 9, 12, 24, 36, 48, 60]
    cva_charges = []

    for frequency in collateral_frequencies:
        cva_charge = calculate_total_cva_with_collateral(epe_profile, hazard_rates, lgd,
                                                         risk_free_rate, T, dt, frequency)
        cva_charges.append(cva_charge)

    print("\n4a) CVA with Different Collateral Frequencies:")
    for freq, cva in zip(collateral_frequencies, cva_charges):
        print(f"  {freq:3d} months: {cva:8.2f} (reduction: {100 * (1 - cva / cva_netted_portfolio):5.1f}%)")

    # Initial margin analysis
    initial_margins = [1e6, 10e6, 100e6]
    cva_with_initial_margins = []

    for initial_margin in initial_margins:
        cva_charge = calculate_total_cva_with_initial_margin(epe_profile, hazard_rates, lgd,
                                                             risk_free_rate, T, dt, initial_margin)
        cva_with_initial_margins.append(cva_charge)

    print("\n4b) CVA with Initial Margin:")
    print(f"No margin: {cva_netted_portfolio:.2f}")
    for margin, cva in zip(initial_margins, cva_with_initial_margins):
        print(f"{margin / 1e6:5.0f}M EUR: {cva:8.2f} (reduction: {100 * (1 - cva / cva_netted_portfolio):5.1f}%)")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: EPE Profile
    months = np.arange(len(discounted_epe_profile))
    axes[0, 0].plot(months, discounted_epe_profile, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (months)')
    axes[0, 0].set_ylabel('Discounted EPE (EUR)')
    axes[0, 0].set_title('Monthly Expected Positive Exposure Profile')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Collateral Frequency Impact
    axes[0, 1].plot(collateral_frequencies, cva_charges, 'go-', linewidth=2, markersize=8, label='With Collateral')
    axes[0, 1].axhline(y=cva_netted_portfolio, color='r', linestyle='--', linewidth=2, label='No Collateral')
    axes[0, 1].set_xlabel('Collateral Posting Frequency (months)')
    axes[0, 1].set_ylabel('CVA Charge (EUR)')
    axes[0, 1].set_title('Impact of Collateral Frequency on CVA')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Initial Margin Impact
    margin_labels = ['No Margin'] + [f"{x / 1e6:.0f}M" for x in initial_margins]
    margin_values = [cva_netted_portfolio] + cva_with_initial_margins
    axes[1, 0].bar(margin_labels, margin_values, color=['red', 'orange', 'yellow', 'green'])
    axes[1, 0].set_xlabel('Initial Margin (EUR)')
    axes[1, 0].set_ylabel('CVA Charge (EUR)')
    axes[1, 0].set_title('Impact of Initial Margin on CVA')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: Parameter Sensitivity
    parameters = ['Base\n(15% vol, 80% corr)', 'High Vol\n(30% vol, 80% corr)', 'Low Corr\n(15% vol, 40% corr)']
    cva_values = [cva_netted_portfolio, cva_netted_3a, cva_netted_3b]
    colors = ['blue', 'red', 'green']
    axes[1, 1].bar(parameters, cva_values, color=colors)
    axes[1, 1].set_ylabel('CVA Charge (EUR)')
    axes[1, 1].set_title('CVA Sensitivity to Market Parameters')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 80)
    print("CVA ANALYSIS COMPLETE")
    print("=" * 80)