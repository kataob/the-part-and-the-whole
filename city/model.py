"""
Model 2 — the fake city.

Not an optimisation: a behavioural response. Given hourly electricity
prices (and, reserved for later, carbon intensity), each of four
building archetypes decides -- hour by hour -- what share of its heat
demand to meet with a heat pump vs a gas boiler, via a smooth logistic
share function. Output is the city's hourly electricity demand, which
feeds back into Model 1.

Interpretation: the hour-by-hour share is a stand-in for a population
of dual-fuel / flexible buildings responding to the price signal in
real time, not a one-off annual investment choice. That's what makes
the response continuous and price-hour-resolved, matching what the
national model can actually hand back each iteration.

Time grid: must match the national model's (day, hour) index -- same
`days`/`hours` as national.model.NationalModelData. In the coupled loop
these should come from one shared time-grid object; here they're
duplicated so this module can be built/tested standalone.
"""

import math


class CityModelData:
    def __init__(self):
        # --- time structure (must match the national model's grid) ---
        self.days = [0, 1, 2, 3]
        self.hours = list(range(24))

        # --- building stock ---
        # share: fraction of total city floor area
        # specific_heat_loss_W_per_m2K: envelope heat loss per m2 per
        #   degree K -- high for leaky old stock, low for new/insulated
        self.archetypes = {
            "old_detached":  {"share": 0.25, "specific_heat_loss_W_per_m2K": None},
            "new_detached":  {"share": 0.20, "specific_heat_loss_W_per_m2K": None},
            "old_apartment": {"share": 0.35, "specific_heat_loss_W_per_m2K": None},
            "new_apartment": {"share": 0.20, "specific_heat_loss_W_per_m2K": None},
        }
        assert abs(sum(a["share"] for a in self.archetypes.values()) - 1.0) < 1e-9

        # Total floor area is THE knob for city size. Set this so peak
        # city demand lands at 15-25% of national peak demand -- per
        # the brief, that's the difference between a project with a
        # result and a project with a shrug.
        self.total_floor_area_m2 = None  # TODO

        self.indoor_setpoint_C = 20.0

        # --- non-heating baseline demand, MW, per (day, hour) ---
        # lighting/appliances/etc: independent of price and temperature
        self.baseline_demand_MW = {}  # TODO: e.g. flat + a daily occupancy shape

        # --- weather: MUST be the same realisation the national model's
        # wind/solar generator uses, so cold snaps and low wind line up ---
        self.outdoor_temp_C = {}  # TODO: {(d, h): float}

        # --- heat pump / boiler economics ---
        self.gas_price_per_MWh_fuel = None   # $/MWh of gas commodity (fixed, not hourly)
        self.boiler_efficiency = 0.90
        # COP is piecewise-linear between these two anchors, then
        # clamped at the floor -- a heat pump doesn't stop working
        # below -5C, it just doesn't keep getting worse forever.
        self.cop_anchor_high = (7.0, 3.5)    # (outdoor_C, COP)
        self.cop_anchor_low = (-5.0, 2.0)
        self.cop_floor = 1.3

        # Logistic steepness for the heat-pump/boiler share function.
        # Higher = sharper response to price signals (approaches a hard
        # switch, which the brief specifically warns kills convergence);
        # lower = mushier but easier for the outer loop to damp.
        self.share_steepness = 8.0


def heat_pump_cop(outdoor_temp_C: float, data: CityModelData) -> float:
    """Piecewise-linear COP(outdoor temperature), clamped at the floor."""
    (t_hi, cop_hi), (t_lo, cop_lo) = data.cop_anchor_high, data.cop_anchor_low
    slope = (cop_hi - cop_lo) / (t_hi - t_lo)
    if outdoor_temp_C >= t_hi:
        cop = cop_hi + slope * (outdoor_temp_C - t_hi)
    elif outdoor_temp_C <= t_lo:
        cop = cop_lo + slope * (outdoor_temp_C - t_lo)
    else:
        cop = cop_lo + (outdoor_temp_C - t_lo) * slope
    return max(cop, data.cop_floor)


def heat_pump_share(cost_hp_per_MWh_heat: float, cost_gas_per_MWh_heat: float, steepness: float) -> float:
    """
    Smooth logistic share of heat demand met by heat pumps.

    x > 0 when the heat pump is cheaper than the boiler; the share
    slides continuously through 0.5 at cost parity instead of snapping
    between 0 and 1. That continuity is what lets the outer loop damp
    into a fixed point instead of bouncing between two corner solutions
    every iteration -- see the brief's note on hard switches.
    """
    denom = max(cost_gas_per_MWh_heat, 1e-6)
    x = (cost_gas_per_MWh_heat - cost_hp_per_MWh_heat) / denom
    y = steepness * x
    # numerically stable sigmoid -- a spiky dual price (early loop
    # iterations, or a near-degenerate LP) can make y large enough that
    # exp() overflows before it's clamped by the 0/1 saturation anyway.
    if y >= 0:
        return 1.0 / (1.0 + math.exp(-y))
    ey = math.exp(y)
    return ey / (1.0 + ey)


def archetype_heat_loss_coeff_kW_per_K(name: str, data: CityModelData) -> float:
    """Aggregate heat loss coefficient for one archetype's share of the city stock, kW/K."""
    a = data.archetypes[name]
    archetype_floor_area_m2 = data.total_floor_area_m2 * a["share"]
    return a["specific_heat_loss_W_per_m2K"] * archetype_floor_area_m2 / 1000.0


def solve_city(prices: dict, carbon_intensity: dict, data: CityModelData) -> dict:
    """
    Given hourly national prices, compute the city's hourly electricity
    demand.

    Parameters
    ----------
    prices : {(d, h): $/MWh}                from national.model.solve_national()["price"]
    carbon_intensity : {(d, h): tCO2/MWh}    currently unused in the demand
        calculation -- kept in the signature so it's available for a
        future emissions-aware extension (e.g. carbon-weighted retrofit
        incentives) without changing the loop's call site.
    data : CityModelData

    Returns
    -------
    demand : {(d, h): MWh} -- same (day, hour) keys as `prices`.
    """
    heat_loss_coeff = {
        name: archetype_heat_loss_coeff_kW_per_K(name, data)
        for name in data.archetypes
    }
    cost_gas_per_MWh_heat = data.gas_price_per_MWh_fuel / data.boiler_efficiency

    demand = {}
    for (d, h), price in prices.items():
        outdoor_temp = data.outdoor_temp_C[(d, h)]
        cop = heat_pump_cop(outdoor_temp, data)

        cost_hp_per_MWh_heat = price / cop
        hp_share = heat_pump_share(cost_hp_per_MWh_heat, cost_gas_per_MWh_heat, data.share_steepness)

        hp_electricity_MW = 0.0
        for name, coeff_kW_per_K in heat_loss_coeff.items():
            heat_needed_kW = max(coeff_kW_per_K * (data.indoor_setpoint_C - outdoor_temp), 0.0)
            heat_needed_MW = heat_needed_kW / 1000.0
            hp_electricity_MW += heat_needed_MW * hp_share / cop

        baseline_MW = data.baseline_demand_MW.get((d, h), 0.0)
        demand[(d, h)] = hp_electricity_MW + baseline_MW  # MW over 1h == MWh

    return demand


if __name__ == "__main__":
    # Smoke test with placeholder numbers -- replace with real
    # parameters.yaml + shared weather series before trusting any output.
    data = CityModelData()
    for name in data.archetypes:
        data.archetypes[name]["specific_heat_loss_W_per_m2K"] = 3.5 if "old" in name else 1.2
    data.total_floor_area_m2 = 8_000_000  # placeholder city size
    data.gas_price_per_MWh_fuel = 40.0
    data.baseline_demand_MW = {(d, h): 50.0 for d in data.days for h in data.hours}
    data.outdoor_temp_C = {
        (d, h): 5.0 - 10.0 * (d == 0) + 3.0 * math.sin(h / 24 * 2 * math.pi)
        for d in data.days for h in data.hours
    }

    # fake price signal: expensive in the evening, cheap overnight
    prices = {
        (d, h): 30.0 + 100.0 * math.exp(-((h - 18) ** 2) / 8.0)
        for d in data.days for h in data.hours
    }
    carbon = {(d, h): 0.3 for d in data.days for h in data.hours}

    demand = solve_city(prices, carbon, data)
    print("Sample demand (day 0), every 3h:", {h: round(demand[(0, h)], 1) for h in range(0, 24, 3)})
    print("Total demand, MWh:", round(sum(demand.values()), 1))
