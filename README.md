# SW_summerschool

Exercises for a computational methods summer school, based around a shallow-water model.

## Install locally

From this directory, using Python 3.9 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Editable installation makes changes under `src/` available without reinstalling.
To include the notebook tools, use `python -m pip install -e ".[notebooks]"`.
Use that same environment as the notebook kernel.

## Run a short simulation

From the repository root:

```python
from sw_summerschool import SW_model

model = SW_model("examples/cfg_rossby.yaml", outfile="rossby_demo.nc")
try:
    model.integrate(20)
finally:
    model.io.close()
```

Configuration paths and output paths are relative to the current working directory.
The model overwrites an existing output file of the same name, so choose a new name
when preserving previous results. Import utilities using, for example,
`from sw_summerschool import plotting` or
`from sw_summerschool.arakawa_c import ArakawaCGrid`.
The old top-level imports such as `from driver import SW_model` have been replaced
by package imports; the existing notebook imports have been updated.

## Layout

- `src/sw_summerschool/`: model, integration, initial conditions, IO, plotting and helpers.
- `examples/`: original notebooks, YAML configurations, and preserved local NetCDF results.
- `tests/`: short integration and output checks.

Launch `jupyter lab` from `examples/` to retain the notebooks' relative file paths.
The notebooks are exploratory work, with their existing outputs retained; they are
not guaranteed to run from top to bottom in a fresh kernel. In particular, some
cells reference `model` or `ds` before creating them, delete output files, or enable
LaTeX rendering (which requires a separate LaTeX installation).
Generated NetCDF files in `examples/` are ignored by Git and are not shipped in the
Python wheel. The examples remain in the source checkout.

## Existing configuration limitations

The original configurations are preserved:

- `cfg.yaml` selects `barotropic_jet` with `Ro: 0`, which divides by zero during
  initialization. Supply a nonzero override, e.g. `Ro=0.25`, for that experiment.
- `cfg_gravwaves.yaml` uses an older schema (`lam`, a string `ic`, and no `beta`).
  To run it with the current model, explicitly supply `Ro=0`, `beta=0`, and
  `ic={"type": "plane_gravity_wave", "amp": 0.001, "width": 0.4}` when constructing `SW_model`.

## Validation and cleanup scope

```sh
python -m unittest discover -s tests -v
```

The reorganization preserves the numerical equations, boundary-condition code,
initial-condition functions, and AB3/RK4 integration routines. Minimal existing
startup/IO errors were corrected: `self.interptype`, instance references for edge
metadata and printed diagnostics, and NetCDF dimension names matching the writer.
Eight-step periodic and wall-boundary simulations were compared against the
original modules with those same repairs applied, using N=8 and Ro=0.25. All
state arrays and saved NetCDF arrays matched exactly. The unmodified original
could not execute because of its syntax error.
