# Neuro-Cognitive Multivel Causal Modeling (NC-MCM)

This repository provides the [Julia](https://julialang.org/) code for learning and the Python code for visualizing neuro-cognitive multilevel causal models (NC-MCMs), as described in [M. Grosse-Wentrup, A. Kumar, A. Meunier, and M. Zimmer, Neuro-Cognitive Multilevel Causal Modeling: A Framework that Bridges the Explanatory Gap between Neuronal Activity and Cognition](addlink). It also provides the scripts for reproducing the results described in the original paper.

## Setting up the `nilab` toolbox

The code for learning a NC-MCM is implemented in the `nilab` toolbox, which is part of this repository. To use the `nilab` toolbox, start Julia, switch to `Pkg` mode with `]`, activate the NC-MCM environment via `activate /path/to/nc-mcm/nilab/`, and then import `nilab` via `using nilab`.

## Learning a neuro-cognitive causal model

The main function in `nilab` to learn a NC-MCM is `learn_mcm()` (located in `../ncmcm/nilab/src/mcm.jl`): 

```
This function learns a multi-level causal model from micro-level states x and behaviors y:

function mcm(x, b ; dimreduction = [], predmodel = logreg, clustering = kmeans, markov_test = markovian)

Input:

    x   neuronal data [samples x features]
    b   discrete behavioral labels [samples]

It relies on the following processing steps:

    1. Applies dimensionality reduction method `dimreduction` to input data x [none implemented as of now]
    2. Uses prediction model `predmodel` to predict the probabilities of each behavior at every sample of x [logistic regression]
    3. Clusters the predicted probabilities using the `clustering` method to generate macroscopcic states [kmeans]
    4. Uses the test given by `markov_test` to test H0: th macroscopic state transitions form a Markov chain

It returns a mcm model with the following elements:

    x       raw data
    K       number of cognitive states
    c       cognitive states
    p       p-values for rejecting H0: Markov chain for each k in 1:K
    b       behavioral labels
    bpred   predicted behavioral labels
    bprob   predicted behavioral probabilities
```

The four processing steps call further methods implemented in the `nilab` toolbox. The choice of algorithms is currently limited, but new methods for each of the processing steps can be implemented in `nilab` and then called from `learn_mcm()`.

## Reproducing the results in the original [NC-MCM paper](addlink)

After setting up the `nilab` toolbox in Julia as described above:

1. Download the experimental data from [https://osf.io/2395t/](https://osf.io/2395t/) and place the file `WT_NoStim.mat` in the main `nc-mcm` directory.
2. In Julia and after importing the `nilab` toolbox, run `include('script_to_generate_ncmcms.jl')` (this may take a while).

This will store the results in individual (`.jld2` and `.npz`) files for each worm in the current directory. The `.npz` files are used for plotting the results in Python. The original files used in the publication are provided in `precomputed_results/`.

To plot the results, and assuming you use [Conda](https://docs.conda.io/en/latest/) as the package management environment, create a virtual environment with the provided `.yml` file: `conda env create -f ncmcm.yml`. You can then reproduce the plots in the original paper in Python by, first, importing the plotting functions via `from plot_ncmcms import *` in `../nc-mcm/plotting/` and, second, running the plotting script in the current namespace: `run -i script_to_plot_results.py`.