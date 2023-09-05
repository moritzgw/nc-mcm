
using Random
using Statistics
using Flux
using Optim

export logreg, train!, predict, loss, test, predictprob

## model struct

mutable struct logreg

	# model parameters
	labels 		# names of binary labels
	m			# number of samples
	weights
	intercept
	loss_train
	l1penalty
	l2penalty

	# optimization parameters
	# eta0
	# tol
	# max_steps
	max_attempts
	success

	# initial values
	logreg() = (z = new();
		z.labels = [];
		z.m = [];
		z.weights = [];
		z.intercept = [];
		z.loss_train = [];
		z.l1penalty = 0.0;
		z.l2penalty = 0.0;
		# z.eta0 = 0.1;
		# z.tol = 1e-6;
		# z.max_steps = 1e5 ;
		z.max_attempts = 10;
		z.success = [];
		return z )

end

## train

function train!(model::logreg, data, labels)

	# initialize model (if necessary)
	m, N = size(data)

	if isempty(model.intercept)

		model.weights = randn(N)
		model. intercept = randn(1)

	end

	# check labels for two classes and relabel
	label_names = unique(labels)
	
	if length(label_names) > 2

		error("More than two unique elements in labels can not be handled by the binary logreg model!")

	end

	i0 = findall(labels .== label_names[1])
	i1 = findall(labels .== label_names[2])

	y = zeros(length(labels))
	y[i0] .= -1.0
	y[i1] .= 1.0

	# append intercept ones to data
	x = cat(data, ones(m, 1), dims = 2)
	w0 = cat(model.weights, model.intercept, dims = 1)
	w = []

	# train model
	attempt = 1
	success = false
	# eta0 = model.eta0
	# loss_after_training = NaN

	l1 = model.l1penalty
	l2 = model.l2penalty

	f = z -> trainloss(z, x, y, l1, l2)

	while (attempt <= model.max_attempts) & (success == false) 

		optim_result = optimize(f, w0, Optim.BFGS())
		success = Optim.converged(optim_result)
		w = Optim.minimizer(optim_result)
		w0 = randn(size(w0))
		attempt += 1

	end

	# store parameters
	model.success = success
	model.m = m 
	model.labels = label_names
	model.weights = vec(w[1:end-1])
	model.intercept = w[end]

	# compute training error
	ypred = predict(model, data)
	loss_train = loss(model, ypred, labels)
	model.loss_train = loss_train

end

## predict

function predict(model::logreg, data)

	w = model.weights
	b = model.intercept

	y_pred = Int.((1 ./ (1 .+ exp.(-data*w .- b) )) .>= 0.5)

	i0_pred = findall(y_pred .== 0)
	i1_pred = findall(y_pred .== 1)

	y_pred[i0_pred] .= model.labels[1]
	y_pred[i1_pred] .= model.labels[2]

	return(y_pred)

end

function predictprob(model::logreg, data)

	w = model.weights
	b = model.intercept

	prob_pred = 1 ./ (1 .+ exp.(-data*w .- b) )

	return(prob_pred)

end

## loss for logreg

function loss(model::logreg, ypred, y)

	acc = mean(ypred .== y)

end

## trainloss for logreg

function trainloss(w, data, labels, l1, l2)

	regret = 0.0

	if (l1 > 0) | (l2 > 0)

		lambda0 = 1 - l1 - l2
		lambda1 = l1
		lambda2 = l2

		regret = lambda0 .* mean( log.(1 .+ exp.(-labels.*(data*w)))) + lambda1 .* sum(abs.(w)) + lambda2 .* sum(w.^2)

		# println(lambda0, lambda1, lambda2, regret)

	else

		regret = mean( log.(1 .+ exp.(-labels.*(data*w))))

	end

	return regret

end

## test function

function test(model::logreg, offset = [1 1])

	# set parameters
	m = 1000
	N = 2
	sigma = 2.5

	# create data
	m2 = convert(Int64, m/2)
	labels = cat(zeros(m2), ones(m2), dims = 1);
	data0 = sigma .* randn(m2, N) + repeat(offset, outer = [m2, 1]);
	data1 = sigma .* randn(m2, N) - repeat(offset, outer = [m2, 1]);
	data = cat(data0, data1, dims = 1)

	# train model and predict
	train!(model, data, labels)
	y_pred = predict(model, data)
	acc = loss(model, y_pred, labels)
	println("Train: ", acc) 

	return(model)

end
