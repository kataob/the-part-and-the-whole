"""
Assembles a runnable scenario from parameters.yaml + the synthetic
weather generator: a NationalModelData, a CityModelData, and the fixed
rest-of-country demand profile the city sits alongside.

This is the single source of truth for "what does the dummy world look
like" -- national/model.py, city/model.py and coupling/loop.py all had
their own ad hoc placeholder data in earlier iterations; this replaces
all of that so the two models share one consistent weather realisation
and cost basis, per parameters.yaml.
"""

import math
from pathlib import Path

import yaml

from city.model import CityModelData, heat_pump_cop
from common.weather import generate_weather
from national.model import NationalModelData

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_parameters(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "parameters.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _annualise(tech_params: dict, discount_rate: float) -> float:
    """CAPEX ($/MW) + fixed O&M ($/MW/year) -> annualised cost ($/MW/year)."""
    life = tech_params["economic_life_years"]
    crf = discount_rate * (1 + discount_rate) ** life / ((1 + discount_rate) ** life - 1)
    return tech_params["capex_per_MW"] * crf + tech_params["fixed_om_per_MW_year"]


def build_time_grid(params: dict):
    """
    Returns (days, hours, day_of_year, day_weight) built from the
    `representative_days:` block, in the order the seasons are listed
    there (so day id 0 = first season listed, etc).
    """
    season_names = list(params["representative_days"].keys())
    days = list(range(len(season_names)))
    hours = list(range(24))
    day_of_year = {i: params["representative_days"][name]["day_of_year"] for i, name in enumerate(season_names)}
    day_weight = {i: params["representative_days"][name]["real_days"] for i, name in enumerate(season_names)}

    total_days = sum(day_weight.values())
    if total_days != 365:
        raise ValueError(f"representative_days real_days must sum to 365, got {total_days}")

    return days, hours, day_of_year, day_weight


def _diurnal_national_shape(hour: int, p: dict) -> float:
    """Double-humped morning/evening peak shape, unnormalised."""
    sigma = 2.5
    return (
        0.55
        + 0.35 * math.exp(-((hour - p["morning_peak_hour"]) ** 2) / (2 * sigma**2))
        + 0.45 * math.exp(-((hour - p["evening_peak_hour"]) ** 2) / (2 * sigma**2))
    )


def build_rest_of_country_demand(params: dict, day_of_year: dict, days: list, hours: list) -> dict:
    """
    Fixed synthetic national load shape for everything OUTSIDE the city
    -- not behaviourally modeled, just a documented seasonal + diurnal
    shape. Higher in winter (heating elsewhere), double-humped within
    the day.
    """
    p = params["national"]["rest_of_country"]
    peak_warm_day = params["weather"]["temperature"]["peak_warm_day_of_year"]

    shape_vals = [_diurnal_national_shape(h, p) for h in hours]
    mean_shape = sum(shape_vals) / len(shape_vals)

    demand = {}
    for d in days:
        doy = day_of_year[d]
        # 0 at the warmest day of the year, 1 at the coldest
        season_frac = 0.5 * (1 - math.cos(2 * math.pi * (doy - peak_warm_day) / 365.0))
        day_avg = p["average_demand_MW"] * (1 + (p["winter_summer_demand_ratio"] - 1) * season_frac)
        for h in hours:
            demand[(d, h)] = day_avg * (shape_vals[h] / mean_shape)
    return demand


def build_national_data(params: dict, weather: dict, days: list, hours: list,
                         day_weight: dict, rest_of_country_demand: dict) -> NationalModelData:
    data = NationalModelData()
    data.days, data.hours, data.day_weight = days, hours, day_weight

    techs = params["national"]["technologies"]
    discount_rate = params["national"]["discount_rate"]
    dispatch_and_vre = ["gas", "wind", "solar"]

    data.capex = {name: _annualise(techs[name], discount_rate) for name in dispatch_and_vre}
    data.var_cost = {name: techs[name]["var_cost_per_MWh"] for name in dispatch_and_vre}
    data.emission_factor = {name: techs[name]["emission_factor_tCO2_per_MWh"] for name in dispatch_and_vre}

    battery = techs["battery"]
    data.batt_capex = _annualise(battery, discount_rate)
    data.batt_charge_eff = battery["charge_efficiency"]
    data.batt_discharge_eff = battery["discharge_efficiency"]
    data.batt_duration_hours = battery["duration_hours"]

    data.co2_cap_tonnes = params["national"]["co2_cap_tonnes"]

    data.avail = {}
    for (d, h), cf in weather["wind_cf"].items():
        data.avail[("wind", d, h)] = cf
    for (d, h), cf in weather["solar_cf"].items():
        data.avail[("solar", d, h)] = cf

    # Sensible default so the national model is runnable standalone;
    # the coupling loop overwrites this every iteration with
    # rest_of_country + city demand.
    data.demand = dict(rest_of_country_demand)

    return data


def _build_city_baseline_demand(total_floor_area_m2: float, params: dict, days: list, hours: list) -> dict:
    p = params["city"]["baseline_non_heating"]
    base_MW = p["per_m2_W"] * total_floor_area_m2 / 1e6   # W/m2 * m2 = W -> MW
    uplift = p["occupied_hours_uplift"]
    trough_factor = 2.0 - uplift  # keeps day/night roughly centred on 1x

    demand = {}
    for d in days:
        for h in hours:
            factor = uplift if 7 <= h <= 22 else trough_factor
            demand[(d, h)] = base_MW * factor
    return demand


def _calibrate_total_floor_area_m2(params: dict, weather: dict, rest_of_country_demand: dict) -> float:
    """
    Back-of-envelope sizing so the city lands near
    `target_share_of_national_peak` of national peak demand.

    Assumes near-full heat-pump uptake at the single coldest hour in
    the weather set (a conservative/simple worst case, not the actual
    equilibrium share). This is a starting point, not a solved
    calibration -- after running the coupling loop, check the actual
    converged city/national split and adjust `target_share_of_national_peak`
    or the archetype/baseline figures if it's landed outside 15-25%.
    """
    city_p = params["city"]
    target_share = city_p["target_share_of_national_peak"]
    rest_peak = max(rest_of_country_demand.values())
    target_city_peak_MW = target_share / (1 - target_share) * rest_peak

    coldest_key = min(weather["temperature_C"], key=weather["temperature_C"].get)
    coldest_temp = weather["temperature_C"][coldest_key]

    archetypes = city_p["archetypes"]
    avg_heat_loss_W_per_m2K = sum(a["share"] * a["specific_heat_loss_W_per_m2K"] for a in archetypes.values())
    delta_T = max(city_p["indoor_setpoint_C"] - coldest_temp, 0.0)
    heat_demand_per_m2_kW = avg_heat_loss_W_per_m2K * delta_T / 1000.0

    hp_p = city_p["heat_pump"]
    dummy_city_data = CityModelData()
    dummy_city_data.cop_anchor_high = tuple(hp_p["cop_anchor_high"])
    dummy_city_data.cop_anchor_low = tuple(hp_p["cop_anchor_low"])
    dummy_city_data.cop_floor = hp_p["cop_floor"]
    cop_at_coldest = heat_pump_cop(coldest_temp, dummy_city_data)
    elec_per_m2_kW = heat_demand_per_m2_kW / cop_at_coldest

    baseline_p = city_p["baseline_non_heating"]
    baseline_peak_per_m2_kW = baseline_p["per_m2_W"] * baseline_p["occupied_hours_uplift"] / 1000.0

    peak_demand_per_m2_MW = (elec_per_m2_kW + baseline_peak_per_m2_kW) / 1000.0
    return target_city_peak_MW / peak_demand_per_m2_MW


def build_city_data(params: dict, weather: dict, days: list, hours: list,
                     total_floor_area_m2: float) -> CityModelData:
    data = CityModelData()
    data.days, data.hours = days, hours

    for name, a in params["city"]["archetypes"].items():
        data.archetypes[name] = {
            "share": a["share"],
            "specific_heat_loss_W_per_m2K": a["specific_heat_loss_W_per_m2K"],
        }
    data.total_floor_area_m2 = total_floor_area_m2
    data.indoor_setpoint_C = params["city"]["indoor_setpoint_C"]
    data.baseline_demand_MW = _build_city_baseline_demand(total_floor_area_m2, params, days, hours)
    data.outdoor_temp_C = weather["temperature_C"]

    hp_p = params["city"]["heat_pump"]
    data.cop_anchor_high = tuple(hp_p["cop_anchor_high"])
    data.cop_anchor_low = tuple(hp_p["cop_anchor_low"])
    data.cop_floor = hp_p["cop_floor"]
    data.share_steepness = hp_p["share_steepness"]

    gas_p = params["city"]["gas_boiler"]
    data.gas_price_per_MWh_fuel = gas_p["price_per_MWh_fuel"]
    data.boiler_efficiency = gas_p["efficiency"]

    return data


def build_scenario(params_path: str | Path | None = None) -> dict:
    """
    One-call scenario builder.

    Returns
    -------
    dict with keys: 'params', 'weather', 'days', 'hours', 'day_weight',
    'national_data', 'city_data', 'rest_of_country_demand'.
    """
    params = load_parameters(params_path)
    days, hours, day_of_year, day_weight = build_time_grid(params)

    weather = generate_weather(days, hours, day_of_year, params["weather"], params["random_seed"])
    rest_of_country_demand = build_rest_of_country_demand(params, day_of_year, days, hours)
    total_floor_area_m2 = _calibrate_total_floor_area_m2(params, weather, rest_of_country_demand)

    national_data = build_national_data(params, weather, days, hours, day_weight, rest_of_country_demand)
    city_data = build_city_data(params, weather, days, hours, total_floor_area_m2)

    return {
        "params": params,
        "weather": weather,
        "days": days,
        "hours": hours,
        "day_weight": day_weight,
        "national_data": national_data,
        "city_data": city_data,
        "rest_of_country_demand": rest_of_country_demand,
    }


if __name__ == "__main__":
    scenario = build_scenario()
    rest_peak = max(scenario["rest_of_country_demand"].values())
    city_baseline_peak = max(scenario["city_data"].baseline_demand_MW.values())
    coldest = min(scenario["weather"]["temperature_C"].values())
    windiest_cold_hour = min(
        scenario["weather"]["temperature_C"], key=scenario["weather"]["temperature_C"].get
    )
    print(f"City total floor area: {scenario['city_data'].total_floor_area_m2:,.0f} m2")
    print(f"Rest-of-country peak demand: {rest_peak:,.0f} MW")
    print(f"City baseline (non-heating) peak: {city_baseline_peak:,.0f} MW")
    print(f"Coldest hour: {coldest:.1f} C")
    print(f"Wind CF at coldest hour: {scenario['weather']['wind_cf'][windiest_cold_hour]:.2f} "
          f"(check this is LOW -- that's the anti-correlation working)")
