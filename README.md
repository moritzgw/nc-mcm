# Neuro-Cognitive Multivel Causal Modeling (NC-MCM)

This repository provides the [Julia](https://julialang.org/) code for learning and the Python code for visualizing neuro-cognitive multilevel causal models. It also provides the scripts for reproducing the results described in [Link text Here](https://link-url-here.org).

## Reproducing the results in [Link text Here](https://link-url-here.org)

1. Download the experimental data from [https://osf.io/2395t/](https://osf.io/2395t/) and place the file `WT_NoStim.mat` in the main directory.
2. Start Julia with the `project.toml` file provided with `nilab`: `julia --project=/your_path_to/nc-mcm`
3. Instantiate the project by switching to `Pkg` (via `]`) and calling `instantiate`.
4. Add `nilab` to your path: `push!(LOAD_PATH, "/your_path_to/nc-mcm/nilab/src/")`
5. Import `nilab`: `using nilab`
6. Run the script to generate and store the results (this may take a while): `include('script_to_generate_ncmcms.jl')`



## Learning a neuro-cognitive causal model

## Plotting a neuro-cognitive causal model


## Licenses

