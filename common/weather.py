"""
Synthetic weather shared by both models.

One realisation of temperature/wind/solar feeds both the national
model (wind & solar availability) and the city model (heat demand &
heat-pump COP) -- that sharing is what makes a January evening cold
*and* low-wind *and* high-priced all at once, which is the whole point
of coupling the two models in the first place.

Method, per (representative day, hour):
  temperature = seasonal sinusoid + diurnal sinusoid + shared "regime"
                term + white noise
  wind CF     = seasonal sinusoid (opposite phase to temperature --
                windier in winter) + the SAME shared regime term +
                white noise
  solar CF    = a daylight-hours bump (widens in summer) x a clear-sky
                midday peak shape x cloud noise, zero outside daylight

The shared regime term is a per-day AR(1) process: negative values mean
"cold snap" (colder AND less windy), positive values mean "mild and
breezy". Because temperature and wind both move with the same regime
value, low wind and cold temperatures land together -- the
anti-correlation the brief asks for -- without hand-tuning a
correlation coefficient directly.
"""

import math

import numpy as np


def _ar1(n: int, coefficient: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean, ~unit-variance AR(1) series of length n."""
    x = np.empty(n)
    x[0] = rng.normal(0.0, 1.0)
    innovation_std = math.sqrt(max(1.0 - coefficient**2, 1e-9))
    for i in range(1, n):
        x[i] = coefficient * x[i - 1] + innovation_std * rng.normal(0.0, 1.0)
    return x


def _seasonal_temperature_mean(day_of_year: int, p: dict) -> float:
    return p["annual_mean_C"] + p["seasonal_amplitude_C"] * math.cos(
        2 * math.pi * (day_of_year - p["peak_warm_day_of_year"]) / 365.0
    )


def _diurnal_temperature_offset(hour: int, p: dict) -> float:
    return p["diurnal_amplitude_C"] * math.cos(
        2 * math.pi * (hour - p["diurnal_peak_hour"]) / 24.0
    )


def _seasonal_wind_mean_cf(day_of_year: int, p: dict, peak_warm_day: int) -> float:
    # opposite phase to temperature: minus cos peaks at the COLDEST day
    return p["seasonal_mean_cf"] + p["seasonal_amplitude_cf"] * (
        -math.cos(2 * math.pi * (day_of_year - peak_warm_day) / 365.0)
    )


def _daylight_hours(day_of_year: int, p_solar: dict, peak_warm_day: int) -> float:
    # 1.0 at the warmest/longest day, 0.0 at the opposite point in the year
    season_frac = 0.5 * (1 + math.cos(2 * math.pi * (day_of_year - peak_warm_day) / 365.0))
    lo, hi = p_solar["winter_daylight_hours"], p_solar["summer_daylight_hours"]
    return lo + season_frac * (hi - lo)


def _solar_shape(hour: float, sunrise: float, sunset: float) -> float:
    if hour <= sunrise or hour >= sunset:
        return 0.0
    midpoint = (sunrise + sunset) / 2.0
    half_width = (sunset - sunrise) / 2.0
    return max(math.cos((math.pi / 2.0) * (hour - midpoint) / half_width), 0.0)


def generate_weather(days: list, hours: list, day_of_year: dict,
                      weather_params: dict, seed: int) -> dict:
    """
    Parameters
    ----------
    days, hours : the model's (day, hour) grid, e.g. [0,1,2,3], [0..23]
    day_of_year : {day_id: day_of_year} -- where each representative
        day sits in the seasonal cycle (from parameters.yaml)
    weather_params : the `weather:` block of parameters.yaml
    seed : RNG seed (from parameters.yaml's top-level random_seed)

    Returns
    -------
    dict with 'temperature_C', 'wind_cf', 'solar_cf', each {(d, h): float}
    """
    temp_p, wind_p, solar_p, regime_p = (
        weather_params["temperature"], weather_params["wind"],
        weather_params["solar"], weather_params["regime"],
    )
    peak_warm_day = temp_p["peak_warm_day_of_year"]

    temperature, wind_cf, solar_cf = {}, {}, {}

    for d in days:
        # One AR(1) chain per representative day (days aren't temporally
        # contiguous, so each gets its own independent chain), seeded
        # off the global seed for reproducibility.
        day_rng = np.random.default_rng(seed + 1000 * d)
        regime = _ar1(len(hours), regime_p["ar1_coefficient"], day_rng)

        doy = day_of_year[d]
        t_mean_today = _seasonal_temperature_mean(doy, temp_p)
        wind_mean_today = _seasonal_wind_mean_cf(doy, wind_p, peak_warm_day)
        daylight = _daylight_hours(doy, solar_p, peak_warm_day)
        sunrise, sunset = 12.0 - daylight / 2.0, 12.0 + daylight / 2.0

        temp_noise = day_rng.normal(0.0, temp_p["noise_std_C"], size=len(hours))
        wind_noise = day_rng.normal(0.0, wind_p["noise_std_cf"], size=len(hours))
        cloud_noise = day_rng.normal(0.0, solar_p["cloud_noise_std"], size=len(hours))

        for h in hours:
            t = (
                t_mean_today
                + _diurnal_temperature_offset(h, temp_p)
                + temp_p["regime_amplitude_C"] * regime[h]
                + temp_noise[h]
            )
            temperature[(d, h)] = t

            # SAME regime value drives wind: negative regime => colder
            # AND less wind at once (a cold, still snap).
            w = wind_mean_today + wind_p["regime_sensitivity"] * regime[h] + wind_noise[h]
            wind_cf[(d, h)] = float(np.clip(w, 0.0, 1.0))

            s = solar_p["peak_capacity_factor"] * _solar_shape(h, sunrise, sunset)
            s *= max(1.0 - cloud_noise[h], 0.0)
            solar_cf[(d, h)] = float(np.clip(s, 0.0, 1.0))

    return {"temperature_C": temperature, "wind_cf": wind_cf, "solar_cf": solar_cf}
