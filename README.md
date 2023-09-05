# Neuro-Cognitive Multivel Causal Modeling (NC-MCM)

This repository provides the [Julia](https://julialang.org/) code for learning and the Python code for visualizing neuro-cognitive multilevel causal models. It also provides the scripts for reproducing the results described in [Link text Here](https://link-url-here.org).

## Reproducing the results in [Link text Here](https://link-url-here.org)

The Julia `nilab` toolbox implements the functions to learn NC-MCMs. To generate the results in [add link here],

1. download the experimental data from [https://osf.io/2395t/](https://osf.io/2395t/) and place the file `WT_NoStim.mat` in the main directory,
2. start Julia with the `project.toml` file provided with `nilab`: `julia --project=../nc-mcm`,
3. instantiate the project by switching to `Pkg` (via `]`) and calling `instantiate`.
4. add `nilab` to your path: `push!(LOAD_PATH, "../nc-mcm/nilab/src/")`,
5. import `nilab`: `using nilab`,
6. and run the script to generate and store the results (this may take a while): `include('script_to_generate_ncmcms.jl')`.

This will store the results in individual (`.jld2` and `.npz`) files for each worm in the current directory. The `.npz` files are used for plotting the results in Python.

You can then reproduce the plots in [reference to paper] by using the plotting functions in `plotting/plot_ncmcms.py`. The instructions assume that you use [https://docs.conda.io/en/latest/](Conda) as the package management system and [https://ipython.org/](IPython) as your interactive Python environment. They should be easily adaptable to other setups.

1. Create a virtual environment for the plotting functions using the provided `.yml` file: `conda env create -f ncmcm.yml`.
2. Start IPython in this environment and import the plotting functions, i.e., run `from plot_ncmcms import *` in `../nc-mcm/plotting/`.
3. Run the plotting script in the current namespace: `run -i script_to_plot_results.py`.

## Learning a neuro-cognitive causal model

## Plotting a neuro-cognitive causal model

## Licenses

