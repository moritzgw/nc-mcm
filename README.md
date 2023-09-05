# Neuro-Cognitive Multivel Causal Modeling (NC-MCM)

This repository provides the [Julia](https://julialang.org/) code for learning and the Python code for visualizing neuro-cognitive multilevel causal models. It also provides the scripts for reproducing the results described in [Link text Here](https://link-url-here.org).

## Reproducing the results in [Link text Here](https://link-url-here.org)

The Julia `nilab` toolbox implements the functions to learn NC-MCMs. To generate the results in [add link here] using the toolbox,

1. download the experimental data from [https://osf.io/2395t/](https://osf.io/2395t/) and place the file `WT_NoStim.mat` in the main directory,
2. start Julia with the `project.toml` file provided with `nilab`: `julia --project=../nc-mcm`,
3. instantiate the project by switching to `Pkg` (via `]`) and calling `instantiate`.
4. add `nilab` to your path: `push!(LOAD_PATH, "../nc-mcm/nilab/src/")`,
5. import `nilab`: `using nilab`,
6. and run the script to generate and store the results (this may take a while): `include('script_to_generate_ncmcms.jl')`.

This will store the results in individual (`.jld2` and `.npz`) files for each worm in the current directory. The `.npz` files are used for plotting the results in Python.

You can then reproduce the plots in [reference to paper] by using the plotting functions in `plotting/plot_ncmcms.py`. The instructions assume that you use [Conda](https://docs.conda.io/en/latest/) as the package management system and [IPython](https://ipython.org/) as your interactive Python environment. They should be easily adaptable to other setups.

1. Create a virtual environment for the plotting functions using the provided `.yml` file: `conda env create -f ncmcm.yml`.
2. Start IPython in this environment and import the plotting functions, i.e., run `from plot_ncmcms import *` in `../nc-mcm/plotting/`.
3. Run the plotting script in the current namespace: `run -i script_to_plot_results.py`.

## Learning a neuro-cognitive causal model

The main function in `nilab` to learn NC-MCMs is `learn_mcm`:

    ```This function learns a multi-level causal model from micro-level states x and behaviors y:

    function mcm(x, b ; dimreduction = [], predmodel = logreg, clustering = kmeans, markov_test = markovian)

    Input:

        x   neuronal data [smaples x features]
        b   discrete behavioral labels [samples]

    It relies on the following processing steps:

        1. Applies dimensionality reduction method `dimreduction` to input data x [none implemented as of now]
        2. Uses prediction model `predmodel` to predict the probabilities of each behavior at every sample of x
        3. Clusters the predicted probabilities using the `clustering` method to generate macroscopcic states
        4. Uses the test given by `markov_test` to test H0: th macroscopic state transitions form a Markov chain

    It returns a mcm model with the following elements:

        x       raw data
        K       number of cognitive states
        c       cognitive states
        p       p-values for rejecting H0: Markov chain for each k in 1:K
        b       behavioral labels
        bpred   predicted behavioral labels
        bprob   predicted behavioral probabilities```



## Plotting a neuro-cognitive causal model

## Licenses

