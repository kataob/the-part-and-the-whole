"""
Model 1 — the fake country.

A linear program that finds the least-cost mix of generation and storage
capacity to meet demand over a representative-day time slice, subject to
a CO2 cap. Solved with HiGHS.

Time structure: DAYS representative days (default 4, one per season) x
HOURS_PER_DAY hours (default 24) = 96 timesteps, each day weighted by how
many real days it represents (weights should sum to 365).

Hands back to the city:
  - price[t]: dual of the supply=demand constraint, i.e. the marginal
    cost of electricity in that hour ($/MWh).
  - carbon_intensity[t]: emissions per unit of generation that hour
    (tCO2/MWh).
"""

import pyomo.environ as pyo


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class NationalModelData:
    """
    Everything the LP needs, indexed by (day, hour).

    Fill this in from parameters.yaml + synthetic profile generators.
    All the dict/array shapes below are what build_model() expects.
    """

    def __init__(self):
        # --- time structure ---
        self.days = [0, 1, 2, 3]                 # representative day ids
        self.hours = list(range(24))             # hours within a day
        self.day_weight = {0: 91, 1: 91, 2: 91, 3: 92}   # real days represented, sums to 365

        # --- technologies ---
        self.dispatchable = ["gas"]                       # can run any time up to capacity
        self.variable_re = ["wind", "solar"]               # capacity-factor limited
        self.storage = ["battery"]
        self.techs = self.dispatchable + self.variable_re  # techs with a "generation" var

        # --- demand: demand[(d, h)] in MWh ---
        self.demand = {}  # TODO: fill from city baseline + non-electrified heat, MWh per hour

        # --- availability factors, in [0, 1]: avail[(tech, d, h)] ---
        self.avail = {}  # TODO: solar/wind synthetic profiles

        # --- cost parameters ---
        # annualised capex, $/MW/year (already annualised via CRF)
        self.capex = {"gas": None, "wind": None, "solar": None, "battery": None}
        # variable/fuel cost, $/MWh generated
        self.var_cost = {"gas": None, "wind": 0.0, "solar": 0.0}
        # emission factor, tCO2/MWh generated
        self.emission_factor = {"gas": None, "wind": 0.0, "solar": 0.0}

        # --- battery parameters ---
        self.batt_charge_eff = 0.95
        self.batt_discharge_eff = 0.95
        self.batt_duration_hours = 4        # energy capacity = power capacity * duration
        self.batt_capex = None              # $/MW/year, power-based

        # --- policy ---
        self.co2_cap_tonnes = None          # annual cap, sum over weighted hours

        # --- reliability ---
        # Value of lost load: the cost of NOT serving 1 MWh of demand.
        # Without this, a representative-day LP can be forced to build
        # capacity that's fully utilised in only one sample hour, whose
        # dual then has to recover that capacity's ENTIRE annualised
        # cost by itself -- producing scarcity prices in the tens or
        # hundreds of thousands $/MWh. Real systems cap this at their
        # reliability standard's VoLL (order $5,000-10,000/MWh is
        # typical, e.g. ERCOT); this both bounds the price and is the
        # economically correct way to model it, not just a numerical
        # patch.
        self.voll_per_MWh = 8000.0


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(data: NationalModelData) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel(name="national_power_system")

    # -------------------- sets --------------------
    m.D = pyo.Set(initialize=data.days)
    m.H = pyo.Set(initialize=data.hours)
    m.T = m.D * m.H                                   # (day, hour) index

    m.TECH_DISPATCH = pyo.Set(initialize=data.dispatchable)
    m.TECH_VRE = pyo.Set(initialize=data.variable_re)
    m.TECH = m.TECH_DISPATCH | m.TECH_VRE
    m.STORAGE = pyo.Set(initialize=data.storage)

    # -------------------- parameters --------------------
    m.weight = pyo.Param(m.D, initialize=data.day_weight)
    m.demand = pyo.Param(m.T, initialize=data.demand, mutable=True)
    m.avail = pyo.Param(m.TECH_VRE, m.T, initialize=data.avail, mutable=True, default=0.0)

    m.capex = pyo.Param(m.TECH, initialize={k: v for k, v in data.capex.items() if k in data.techs})
    m.var_cost = pyo.Param(m.TECH, initialize={k: v for k, v in data.var_cost.items() if k in data.techs})
    m.emission_factor = pyo.Param(m.TECH, initialize={k: v for k, v in data.emission_factor.items() if k in data.techs})

    m.batt_capex = pyo.Param(initialize=data.batt_capex)
    m.batt_charge_eff = pyo.Param(initialize=data.batt_charge_eff)
    m.batt_discharge_eff = pyo.Param(initialize=data.batt_discharge_eff)
    m.batt_duration = pyo.Param(initialize=data.batt_duration_hours)

    m.co2_cap = pyo.Param(initialize=data.co2_cap_tonnes)
    m.voll = pyo.Param(initialize=data.voll_per_MWh)

    # -------------------- decision variables --------------------
    # capacity to BUILD, MW (or MWh for battery energy, derived from power * duration)
    m.cap = pyo.Var(m.TECH, domain=pyo.NonNegativeReals)
    m.batt_power_cap = pyo.Var(domain=pyo.NonNegativeReals)

    # hourly generation, MWh (== MW average over the hour)
    m.gen = pyo.Var(m.TECH, m.T, domain=pyo.NonNegativeReals)

    # battery operation
    m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals)   # state of charge, MWh

    # unserved demand, MWh -- priced at VoLL in the objective. This is
    # a relief valve, not something you expect to see much of: it
    # bounds the supply_demand dual at VoLL instead of letting a
    # razor-thin capacity margin in one sample hour blow the price up.
    m.unmet_demand = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    # -------------------- constraints --------------------

    # 1) Supply == demand, every hour. This is the constraint whose dual
    #    gives us the electricity price -- keep it as a clean equality,
    #    don't fold anything else into it.
    def supply_demand_rule(m, d, h):
        return (
            sum(m.gen[tech, d, h] for tech in m.TECH)
            + m.discharge[d, h]
            - m.charge[d, h]
            + m.unmet_demand[d, h]
            == m.demand[d, h]
        )
    m.supply_demand = pyo.Constraint(m.D, m.H, rule=supply_demand_rule)

    # 2) Variable renewables: generation capped by capacity x availability
    def vre_limit_rule(m, tech, d, h):
        return m.gen[tech, d, h] <= m.cap[tech] * m.avail[tech, d, h]
    m.vre_limit = pyo.Constraint(m.TECH_VRE, m.D, m.H, rule=vre_limit_rule)

    # 2b) Dispatchable: generation capped by capacity
    def dispatch_limit_rule(m, tech, d, h):
        return m.gen[tech, d, h] <= m.cap[tech]
    m.dispatch_limit = pyo.Constraint(m.TECH_DISPATCH, m.D, m.H, rule=dispatch_limit_rule)

    # 3) Battery state-of-charge accounting.
    #    Each representative day is treated as a self-contained cycle:
    #    SOC wraps around within the day (soc at hour -1 == soc at last hour).
    #    Swap in a rule that carries SOC across days if you want inter-day storage.
    def soc_rule(m, d, h):
        prev_h = data.hours[-1] if h == data.hours[0] else h - 1
        return m.soc[d, h] == (
            m.soc[d, prev_h]
            + m.charge[d, h] * m.batt_charge_eff
            - m.discharge[d, h] / m.batt_discharge_eff
        )
    m.soc_balance = pyo.Constraint(m.D, m.H, rule=soc_rule)

    def soc_capacity_rule(m, d, h):
        return m.soc[d, h] <= m.batt_power_cap * m.batt_duration
    m.soc_capacity = pyo.Constraint(m.D, m.H, rule=soc_capacity_rule)

    def charge_power_rule(m, d, h):
        return m.charge[d, h] <= m.batt_power_cap
    m.charge_power_limit = pyo.Constraint(m.D, m.H, rule=charge_power_rule)

    def discharge_power_rule(m, d, h):
        return m.discharge[d, h] <= m.batt_power_cap
    m.discharge_power_limit = pyo.Constraint(m.D, m.H, rule=discharge_power_rule)

    # 4) CO2 cap on total annual emissions (weighted by representative-day count)
    def co2_cap_rule(m):
        return sum(
            m.weight[d] * m.gen[tech, d, h] * m.emission_factor[tech]
            for tech in m.TECH for d in m.D for h in m.H
        ) <= m.co2_cap
    m.co2_limit = pyo.Constraint(rule=co2_cap_rule)

    # -------------------- objective --------------------
    def total_cost_rule(m):
        capital_cost = (
            sum(m.capex[tech] * m.cap[tech] for tech in m.TECH)
            + m.batt_capex * m.batt_power_cap
        )
        operating_cost = sum(
            m.weight[d] * m.gen[tech, d, h] * m.var_cost[tech]
            for tech in m.TECH for d in m.D for h in m.H
        )
        unmet_cost = sum(
            m.weight[d] * m.unmet_demand[d, h] * m.voll
            for d in m.D for h in m.H
        )
        return capital_cost + operating_cost + unmet_cost
    m.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)

    # -------------------- duals --------------------
    # Set this up on day one, per the brief -- don't discover you need it at 6pm.
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return m


# ---------------------------------------------------------------------------
# Solve + extract results
# ---------------------------------------------------------------------------

def solve_national(data: NationalModelData, solver_name: str = "appsi_highs"):
    """
    Build, solve, and extract prices + carbon intensity.

    Returns
    -------
    result : dict with keys
        'price'            : {(d, h): $/MWh}   -- dual of supply_demand
        'carbon_intensity' : {(d, h): tCO2/MWh}
        'capacity'         : {tech: MW}
        'gen'              : {(tech, d, h): MWh}
        'model'             : the solved pyomo ConcreteModel (for debugging)
    """
    m = build_model(data)

    solver = pyo.SolverFactory(solver_name)
    solver.solve(m, tee=False)

    # The supply_demand constraint is unweighted per representative
    # hour, but the objective's operating cost weights each hour by
    # data.day_weight[d] real days it stands for. So the raw dual comes
    # out in $ per representative-hour-unit -- weight[d] times too
    # large to read as a genuine $/MWh price. Divide it back out.
    price = {
        (d, h): m.dual[m.supply_demand[d, h]] / data.day_weight[d]
        for d in data.days for h in data.hours
    }

    carbon_intensity = {}
    for d in data.days:
        for h in data.hours:
            total_gen = sum(pyo.value(m.gen[tech, d, h]) for tech in m.TECH)
            total_emissions = sum(
                pyo.value(m.gen[tech, d, h]) * data.emission_factor[tech]
                for tech in m.TECH
            )
            carbon_intensity[(d, h)] = total_emissions / total_gen if total_gen > 1e-9 else 0.0

    capacity = {tech: pyo.value(m.cap[tech]) for tech in m.TECH}
    capacity["battery"] = pyo.value(m.batt_power_cap)

    gen = {
        (tech, d, h): pyo.value(m.gen[tech, d, h])
        for tech in data.techs for d in data.days for h in data.hours
    }

    return {
        "price": price,
        "carbon_intensity": carbon_intensity,
        "capacity": capacity,
        "gen": gen,
        "model": m,
    }


if __name__ == "__main__":
    # Smoke test with placeholder numbers -- replace with real
    # parameters.yaml + profile generators before trusting any output.
    data = NationalModelData()
    n_hours = len(data.days) * len(data.hours)

    data.demand = {(d, h): 100.0 for d in data.days for h in data.hours}
    data.avail = {
        (tech, d, h): 0.3 if tech == "wind" else (0.6 if 6 <= h <= 18 else 0.0)
        for tech in data.variable_re for d in data.days for h in data.hours
    }
    data.capex = {"gas": 50_000, "wind": 90_000, "solar": 70_000, "battery": 40_000}
    data.var_cost = {"gas": 60.0, "wind": 0.0, "solar": 0.0}
    data.emission_factor = {"gas": 0.4, "wind": 0.0, "solar": 0.0}
    data.batt_capex = 40_000
    data.co2_cap_tonnes = 1e9  # non-binding placeholder

    result = solve_national(data)
    print("Capacities:", result["capacity"])
    print("Sample prices:", {k: round(v, 2) for k, v in list(result["price"].items())[:5]})
