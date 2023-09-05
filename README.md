# Neuro-Cognitive Multivel Causal Modeling (NC-MCM)

This repository provides the [Julia](https://julialang.org/) code for learning and the Python code for visualizing neuro-cognitive multilevel causal models. It also provides the scripts for reproducing the results described in [Link text Here](https://link-url-here.org).

## Setting up the NC-MCM package

1. Start Julia with the `project.toml` file provided with `nilab`: `julia --project=/your_path_to/nc-mcm`
2. Instantiate the project by switching to `Pkg` (via `]`) and calling `instantiate`.
3. Add `nilab` to your path: `push!(LOAD_PATH, "/your_path_to/nc-mcm/nilab/src/")`
4. Import `nilab`: `using nilab`

## Reproducing the results in [Link text Here](https://link-url-here.org)

1. Download the experimental data from [https://osf.io/2395t/](https://osf.io/2395t/) and place the file `WT_NoStim.mat` in the main directory.
2. In Julia, and after following the instructions above, run the script to generate and store the results (this may take a while): `include('script_to_generate_ncmcms.jl')`



## Learning a neuro-cognitive causal model

## Plotting a neuro-cognitive causal model


## Licenses

