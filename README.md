# the-part-and-the-whole

**Welcome to the Kingdom of Kilowattia.**

It is a small country with a modest amount of wind, a reasonable amount of
sun, one gas plant it would rather not talk about, and a battery. Somewhere
inside it sits the city of **Watt-a-Lot**, which is cold in January, fond of
heat pumps, and (important bit) _much larger_ than Kilowattia's
planners thought it will be.

A city is a whole system and a part of a larger one (which this repo kinda made it literal: two small models, talking to each other until they either agree or give up).

## The premise

**Model 1 is the Kingdom of Kilowattia.** It's a linear program: give it costs, weather,
and a carbon cap, and it tells you the cheapest way to keep enough watts flowing —
how much wind, solar, gas, and battery to build, and at which hour. As a side effect of solving, it also hands an electricity
price for every hour (the dual of supply = demand) and a carbon intensity. The rotating rock the Killowatia is on cannot be safe for its citizens if a lot of carbon flows around.

Kilowattia, however, does not know that Watt-a-Lot is about to change its mind about
heating. To Kilowattia, the city is just a simple demand curve.

**Model 2 is Watt-a-Lot.** It's not an optimiser — it's a population of
buildings behaving. Given the price signal from Kilowattia, each of four
building archetypes decides, hour by hour, what fraction of its heat demand
to serve with a heat pump versus a gas boiler. (Relatively) Simple:

Cheap electricity → more heat pumps. 
Expensive electricity → more gas. 

The decision is a smooth, inert curve, not a Heaviside function, because a city that flips its entire heating system at once in response to a price tick is a cartoon (probably a horrific one).

**The interesting part**: Watt-a-Lot is not a passive customer. It's sized at
~20% of national demand on purpose, so when it electrifies its heating,
Kilowattia's prices actually move. Feed those new prices back into the city,
and it responds again. Feed that new demand back into the country, and prices
move again. (Positive) Loop until it settles, or doesn't.

That loop is twelve lines. Everything else in this repo is what it takes to
give those twelve lines something digestable.

## What's in the box

```
parameters.yaml            every number in this project, with a source and an excuse
common/weather.py          one shared synthetic winter, issued to both models
common/scenario.py         turns parameters.yaml + weather into runnable model inputs
national/model.py          Kilowattia -- Pyomo + HiGHS
city/model.py               Watt-a-Lot -- archetypes, COP curve, logistic heat-pump share
coupling/loop.py            the twelve lines, plus the seeding/logging/sweeping around them
notebooks/walkthrough.ipynb the illustrated version of this README, with plots
```

Run any module directly (`python -m common.scenario`, `python -m
national.model`, `python -m city.model`, `python -m coupling.loop`) and it
solves a smoke-test scenario and prints something sane. That's also the
fastest way to check nothing broke after you change a number.

If you'd rather look at pictures than reconstruct them from this file,
`pixi run notebook` opens [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb)
-- weather plots, the LP's capacity mix and price curves, the heat pump's
COP and share curves, the loop's convergence trajectory, and the
feedback-vs-no-feedback price comparison, all narrated. It's kept in
sync with wherever the loop currently stands, caveats included.

## The one design choice to be mindful of

If Watt-a-Lot were 2% of national load, electrifying its heating would change
Kilowattia's prices by nothing, the loop would converge on iteration two, and
there would be no project — just two models that happen to be able to phone
each other.

So the city is calibrated to roughly 15–25% of national peak demand. Read that
as "a metropolitan region," or as a deliberate stress test. Either is fine, as
long as it's on the record — which is why it's a named, documented knob in
`parameters.yaml` (`city.target_share_of_national_peak`), and not a number
that merely happened to survive.

## Dispatches from the loop

Mechanically, it works: solve the country, extract prices, run the city, damp,
repeat, log every iteration — because the trajectory is the result, not just
wherever it lands. Along the way, two things were worth fixing rather than
shrugging at:

- **Duals need dividing.** The four representative days each stand in for ~91
  real days, and that weighting only appeared on one side of the model — the
  objective's operating cost, not the hourly balance constraint the price
  comes from. Left alone, that produced electricity prices in the tens of
  thousands of dollars per MWh. That is not a "the grid is having a bad day"
  number. That is a "you forgot to divide by 91" number. Fixed in
  `national/model.py`.
- **Rare-hour scarcity needs a ceiling.** A capacity-expansion LP will happily
  let one razor-thin hour's dual recover an entire technology's annualised
  fixed cost, which is correct in theory and useless in practice. There's now
  a value-of-lost-load slack, so the price can't run away past what a real
  reliability standard would tolerate.

With those fixed, prices land in a believable range: tens of $/MWh, with the
occasional spike into the low thousands.

**What hasn't happened yet: the ping-pong.** Even with damping switched off
entirely, the loop converges in two or three iterations, every time. That's
not a bug — it means that as calibrated, the system is more stable than the
headline result needs it to be. The likeliest culprit is the battery, which is
doing precisely what it was built to do and smoothing away the very price
volatility Watt-a-Lot is supposed to overreact to.

Next session's job is tuning that back in: sharper price swings, a stricter
carbon cap, or a city whose appetite outpaces the battery's ability to flatten
it — until the α-sweep actually shows a boundary between spiralling inward and
ping-ponging forever, instead of everything just working on the first try.

## Weather

All synthetic, all in `parameters.yaml`, all documented with either a source
(NREL ATB 2024, IEA gas prices) or an honest "illustrative" label where there
isn't one.

Temperature, wind, and sun come from one shared generator in
`common/weather.py`, so a cold, still January evening in Kilowattia is a cold,
still January evening in Watt-a-Lot.

## Requirements

With [pixi](https://pixi.sh) (recommended -- locked, reproducible, no
"works on my machine"):

```bash
pixi install
pixi run scenario   # sanity-check the synthetic world
pixi run national   # Kilowattia alone
pixi run city       # Watt-a-Lot alone
pixi run loop       # the actual coupling
```

Without pixi, a plain venv works too:

```bash
python -m venv .venv && source .venv/bin/activate
pip install pyomo highspy pyyaml numpy
python -m coupling.loop
```

HiGHS is free and open-source, so anyone who clones this can actually solve it
(no license server, no "trust me, it works.")