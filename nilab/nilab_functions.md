## nilab Toolbox

. All data variables are assumed to have the samples in the first dimension [samples x features x ...]
. Labels can be any objects and are handled internally

## Dependencies
. Flux
. Combinatorics
. MAT
. Plots
. PlotylBase

## Todo
. Break tie in multiclass prediction randomly
. Add loss functions to base and replace in sub-modules
. Add various encodings to loadeeg()
. Add loading sequentially to loadeeg()

## Loading data

. loadmat() (ok)
. loadeeg() (ok -- but see todos above)
. loadnp() (ok)
. loadbci2000()
. loadpkl()

## Signal processing [nisigproc.jl]

. logbp() (ok)
. cav() (ok)
. hann() (ok)
. standardize()

### base

### model fitting

. splitdata(N, split, random = true) (ok)
. crossvalidate(model, data, folds, split_type) (ok)
. learningcurve(model, data, stepsize, repetitions) (ok)
. sgd() (ok) -- add stochastic sampling, estimate of initial learning rate?
. featureimportance!(model, data, labels)
. parameteroptimization!(model, param, data, labels)

### auxiliary functions

. x, y, ... = splitlabels(data, labels)
. data, labels = fuselabels(x, y)
. z = epoch(x, trigger_times)
. indices = nifind(z, x) (ok)

## Supervised learning functions

### for each model

. train!(model, data, labels)
. predict(model, data)
. loss(model, y_pred, y_true)
. test(model)

### for some models
. trainloss()
. predictprob(mode, data)

## Supervised models

. lda() (ok)
. linreg() (ok)
. logreg() (ok)
. svm
. randomforest

## Unsupervised models

. pca() (ok)
. ica()
. kmeans()
. spectralclustering()

## Causality

. markovian() (ok)
. simulate_markovian (ok)
. vcfl()
. medil()
. sci()

## Statistical tests and model evaluation

. permtest(function, x, y, nperm, two_sided) (ok)
. paired_permtest(function, x, y, nperm, two_sided) (ok)
. confusionmatrix(y_pred, y_true) (ok)
. std(z) (ok)
. cumprob(z) (ok)
. 
. fdr()
. loss01(y_pred, y_true)
. loss_l2(y_pred, y_true)
. loss_l1(y_pred, y_true)
. hsic()
. panova(x, c) (ok)
. fvalue(x, c) (ok)

## Plotting
