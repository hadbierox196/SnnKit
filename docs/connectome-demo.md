# Connectome Disease-Spread Demo: What This Is and Isn't

Companion to `docs/connectome-narrative.md` (the "why does this belong
here" doc) and `notebooks/connectome_disease_spread_demo.ipynb` (the
standalone demo itself). This doc exists to state the scope limits
explicitly, so the demo isn't read as claiming more than it does.

## What it is

- A real, properly-sourced, MIT-licensed connectome (White et al. 1986,
  309 neurons, 2,961 synapses, via OpenWorm's ConnectomeToolbox — see
  `snnkit/connectome/data/SOURCE.md`) loaded directly into snnkit's
  existing sparse graph representation, with zero new infrastructure.
- A linear diffusion (graph heat equation) model of how a scalar
  "pathology load" spreads across that graph over time, validated
  quantitatively against the exact closed-form solution
  (`docs/../tests/test_connectome_diffusion.py`: correlation > 0.999,
  relative L2 error < 2%).
- An interactive visualization you can scrub through time
  (`notebooks/connectome_disease_spread_demo.ipynb`), with a static
  fallback for contexts (e.g. a slide deck) that can't render an
  interactive widget.
- A demonstration that the core engine's sparse-graph substrate
  generalizes past the exact neuron-simulation use case it was built for.

## What it is NOT

- **Not a clinical or biological claim.** The diffusion model is a
  simplified linear ODE (the standard graph heat equation). Real
  prion-like or tau pathology propagation involves nonlinear dynamics
  (production, clearance, saturation, cell-type-specific vulnerability)
  that this model does not attempt to capture. Treat it as "a
  mathematically well-understood dynamical process running on a real
  biological graph," not "a disease progression model."
- **Not validated against real disease-progression data.** The
  validation performed (Week 14) checks the *numerical* correctness of
  the simulation (does the Euler integration match the exact solution to
  the same equation) — it says nothing about whether the *model itself*
  matches any real pathology-spread measurements. No such comparison is
  made or claimed anywhere in this repo.
- **Not a human connectome.** *C. elegans* has 302 neurons; this is not
  a scaled-down human brain, and no claim is made that dynamics here
  transfer to human neurodegenerative disease. It's a real, appropriately
  -scaled, and genuinely interesting graph for demonstrating the engine's
  generality — not a model organism stand-in for human pathology.
- **Not a fundraising claim about a disease-modeling product.** This
  extension exists to demonstrate platform generality (per
  `docs/connectome-narrative.md`), not to position the company as doing
  disease modeling. If a future direction wants to make real biological
  claims, that requires real biological validation data and almost
  certainly domain-expert collaboration — explicitly out of scope for
  what's built here.

## If you want to take this further

- Swap in a nonlinear reaction-diffusion model (production + clearance
  terms) if modeling something closer to real proteinopathy dynamics —
  `snnkit.connectome.diffusion`'s `simulate_diffusion_euler` step function
  is a natural place to add nonlinear terms.
- Swap in a different connectome (e.g. one of the other OpenWorm
  ConnectomeToolbox datasets — Cook et al. 2019, Witvliet et al. 2021 —
  see the toolbox's dataset list) via a new loader function alongside
  `load_white1986_connectome`.
- If pursuing a real biological validation, that's a different project
  phase requiring domain collaborators, not a natural extension of this
  codebase alone.
