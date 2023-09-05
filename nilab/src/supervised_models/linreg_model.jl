## modules used in linreg_model

using Random
using Statistics
using LinearAlgebra

## exports

export linreg, train!, predict, loss, test

## model struct

mutable struct linreg

	# model parameters	
	weights
	intercept
	loss_train

	linreg() = new()

end

## train

function train!(model::linreg, data, labels)

	# nrsamples
	m = size(data)[1]

	# train model
	x = cat(data, ones(m, 1), dims = 2)

	A = x'*x / m
	b = x'*labels / m
	w = pinv(A) * b

	# store parameters
	model.weights = vec(w[1:end-1])
	model.intercept = w[end]

	# compute training error
	ypred = predict(model, data)
	loss_train = loss(model, ypred, labels)
	model.loss_train = loss_train

end

## predict

function predict(model::linreg, data)

	w = model.weights
	b = model.intercept

	y_pred = data*vec(w) .+ b

	return(y_pred)

end

## l2 loss for linreg

function loss(model::linreg, ypred, y)

	return mean( (ypred - y).^2 )

end

## test function

function test(model::linreg)

	# set parameters
	alpha = [1 1]
	beta = 0.5
	sigma = 1
	m = 100

	# create data
	x = randn(m, 2)
	y = x * alpha' .+ beta

	# train model and predict
	train!(model, x, y)
	y_pred = predict(model, x)
	acc = loss(model, y_pred, y)
	println("Train: ", acc) 

	return(model)

end
