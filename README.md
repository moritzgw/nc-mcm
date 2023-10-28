# Neuro-Cognitive Multivel Causal Modeling (NC-MCM)

This repository provides the [Julia](https://julialang.org/) code for learning and the Python code for visualizing neuro-cognitive multilevel causal models (NC-MCMs), as described in [M. Grosse-Wentrup, A. Kumar, A. Meunier, and M. Zimmer, Neuro-Cognitive Multilevel Causal Modeling: A Framework that Bridges the Explanatory Gap between Neuronal Activity and Cognition](https://www.biorxiv.org/content/10.1101/2023.10.27.564404v1). It also provides the scripts for reproducing the results described in the original paper.

## Setting up the `nilab` toolbox

The code for learning a NC-MCM is implemented in the `nilab` toolbox, which is part of this repository. To use the `nilab` toolbox,

1. start Julia, switch to `Pkg` mode with `]`, activate the NC-MCM environment via `activate /path/to/nc-mcm/nilab/`,
2. install all dependencies by calling `instantiate` while still in `Pkg` mode,
3. leave `Pkg` mode and import `nilab` via `using nilab`.

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

## Reproducing the results in the original [NC-MCM paper](https://www.biorxiv.org/content/10.1101/2023.10.27.564404v1)

After setting up the `nilab` toolbox in Julia as described above:

1. Download the experimental data from [https://osf.io/2395t/](https://osf.io/2395t/) and place the file `WT_NoStim.mat` in the main `nc-mcm` directory.
2. In Julia and after importing the `nilab` toolbox, run `include('script_to_generate_ncmcms.jl')` (this may take a while).

This will store the results in individual (`.jld2` and `.npz`) files for each worm in the current directory. The `.npz` files are used for plotting the results in Python. The original files used in the publication are provided in `precomputed_results/`.

The code for plotting the results is written in Python. To recreate the plots, and assuming you use [Conda](https://docs.conda.io/en/latest/) as the package management environment, 

1. create a new Python environment via `conda env create -f ncmcm_plotting.yml`,
2. activate the environment via `conda activate ncmcm_plotting`,
2. in Python, go to the `path/to/nc-mcm/plotting` directory and import the plotting functions `from plot_ncmcms import *`,
3. go to the directory where the `.npz` result files reside and run the plotting script in the current namespace: `run -i script_to_plot_results.py`.

Note that the results obtained by [BunDLe-Net](https://www.biorxiv.org/content/10.1101/2023.08.08.551978v3.abstract), which are used for comparison in the plots for the original manuscript, are provided in `path/to/nc-mcm/precomputed_results/` as `bundlenet_consistent_embedding_worm_X.npz` and need to be copied into the same directory as the `.npz` result files.