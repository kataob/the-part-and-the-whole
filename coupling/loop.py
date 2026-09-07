"""
The coupling loop.

Ping-pongs between the national model (Model 1: cheapest way to meet
demand, plus the price/carbon signal that falls out as a side effect)
and the city model (Model 2: a behavioural demand response to that
price signal) until national prices and city demand stop moving.

The whole contribution is the twelve lines inside run_coupling()'s for
loop. Everything else here is just plumbing: building a consistent
total national demand out of a fixed "rest of country" profile plus
the variable city demand, and seeding + logging the loop.

Sizing note (this is the one design choice that decides whether the
project has a finding, per the brief): `rest_of_country_demand` must be
set so the city is 15-25% of total national load. At 2% the loop
converges on iteration two and there's nothing to show.
"""

import numpy as np

from national.model import NationalModelData, solve_national
from city.model import CityModelData, solve_city


def demand_dict_to_vector(demand: dict, days, hours) -> np.ndarray:
    return np.array([demand[(d, h)] for d in days for h in hours])


def initial_demand_guess(city_data: CityModelData, reference_price: float = 50.0) -> dict:
    """
    Seed the loop with the city's response to a flat, 'typical' price.
    Only needs to be in the right ballpark -- alpha < 1 pulls a bad
    guess in over a few iterations regardless.
    """
    flat_prices = {(d, h): reference_price for d in city_data.days for h in city_data.hours}
    flat_carbon = {(d, h): 0.0 for d in city_data.days for h in city_data.hours}
    return solve_city(flat_prices, flat_carbon, city_data)


def run_coupling(national_data: NationalModelData, city_data: CityModelData,
                  rest_of_country_demand: dict, alpha: float = 0.5,
                  max_iter: int = 30, tol: float = 1e-3, verbose: bool = True):
    """
    Run the damped fixed-point loop between solve_national and solve_city.

    Parameters
    ----------
    national_data, city_data : model configs (weather, costs, etc.) --
        everything EXCEPT the demand being iterated on.
    rest_of_country_demand : {(d, h): MWh} -- fixed, exogenous demand for
        the part of the country that isn't the city. national_data.demand
        is overwritten each iteration with rest_of_country + city demand;
        whatever you set it to beforehand is ignored.
    alpha : damping factor on the update `demand = alpha*new + (1-alpha)*old`.
        1.0 = no damping (take the city's answer outright each time).
    tol : relative L2 error at which to stop.

    Returns
    -------
    history : list of per-iteration dicts (iteration, error,
        city_total_demand_MWh, national_total_demand_MWh, mean_price,
        city_demand_vector) -- this trajectory is the actual result of
        the project, log/plot all of it, not just the final iteration.
    final : {'national_result': ..., 'city_demand': ...} from the last
        iteration run (which may not have converged -- check history).
    """
    days, hours = national_data.days, national_data.hours

    demand = initial_demand_guess(city_data)
    history = []
    national_result = None

    for iteration in range(max_iter):
        # --- the coupling loop (this is the whole point) ---
        national_data.demand = {
            key: rest_of_country_demand[key] + demand[key] for key in demand
        }
        national_result = solve_national(national_data)
        prices, carbon = national_result["price"], national_result["carbon_intensity"]

        demand_new = solve_city(prices, carbon, city_data)

        vec_old = demand_dict_to_vector(demand, days, hours)
        vec_new = demand_dict_to_vector(demand_new, days, hours)
        error = np.linalg.norm(vec_new - vec_old) / np.linalg.norm(vec_old)

        history.append({
            "iteration": iteration,
            "error": error,
            "city_total_demand_MWh": float(vec_new.sum()),
            "national_total_demand_MWh": float(vec_new.sum() + sum(rest_of_country_demand.values())),
            "mean_price": float(np.mean(list(prices.values()))),
            "city_demand_vector": vec_new.copy(),
        })
        if verbose:
            r = history[-1]
            print(f"iter {r['iteration']:2d}  error={r['error']:.4e}  "
                  f"city_demand={r['city_total_demand_MWh']:10.1f} MWh  "
                  f"mean_price={r['mean_price']:8.2f} $/MWh")

        if error < tol:
            break

        demand = {key: alpha * demand_new[key] + (1 - alpha) * demand[key] for key in demand}
        # --- end of the coupling loop ---

    return history, {"national_result": national_result, "city_demand": demand}


def sweep_alpha(national_data: NationalModelData, city_data: CityModelData,
                 rest_of_country_demand: dict, alphas=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
                 max_iter: int = 30, tol: float = 1e-3):
    """
    Run the loop once per damping factor. This is what produces the
    headline figure: total city demand vs. iteration, one line per
    alpha, showing the boundary between oscillation and convergence.

    Each run gets its OWN NationalModelData/CityModelData copy's demand
    state reset implicitly (run_coupling seeds its own initial guess
    and overwrites national_data.demand every iteration), so the same
    data objects can be reused across alphas safely.

    Returns
    -------
    results : {alpha: history} -- pass straight to a plotting routine.
    """
    results = {}
    for alpha in alphas:
        print(f"\n--- alpha = {alpha} ---")
        history, _ = run_coupling(
            national_data, city_data, rest_of_country_demand,
            alpha=alpha, max_iter=max_iter, tol=tol, verbose=True,
        )
        results[alpha] = history
    return results


if __name__ == "__main__":
    from common.scenario import build_scenario

    scenario = build_scenario()
    history, final = run_coupling(
        scenario["national_data"], scenario["city_data"],
        scenario["rest_of_country_demand"], alpha=0.5,
    )
    print(f"\nConverged after {len(history)} iterations, "
          f"final error={history[-1]['error']:.2e}")
