## modules to load

using Random
using Statistics

## exports

export lda, train!, predict, loss, test

## model struct

mutable struct lda

	# data information
	labels 		# names of binary labels
	N::Int64 	# number of features
	m::Int64 	# number of samples


	# model parameters
	weights
	intercept
	bias
	loss_train

	lda() = new()

end

## train

function train!(model::lda, data, labels)

	# model dimenisions
	m, N = size(data)

	# check labels for two classes and relabel
	label_names = unique(labels)
	
	if length(label_names) > 2

		error("More than two unique elements in labels can not be handled by the binary LDA model!")

	end

	# train model
	i0 = findall(labels .== label_names[1])
	i1 = findall(labels .== label_names[2])

	if isempty(i0) || isempty(i1)

		error("Training samples all have the same label.")

	end

	u0 = mean(data[i0, :], dims = 1)
	u1 = mean(data[i1,: ], dims = 1)

	X0 = cov(data[i0, :], dims = 1)
	X1 = cov(data[i1, :], dims = 1)
	X = (X0 + X1) ./ 2

	iX = inv(X)
	w = (u1 - u0) * iX 
	b = 1/2 .* (u0*iX*u0' - u1*iX*u1')

	P0 = length(i0) / m
	P1 = length(i1) / m
	c = log( P0 / P1)

	# store parameters
	model.N = N
	model.m = m 
	model.labels = label_names

	model.weights = w
	model.intercept = b
	model.bias = c

	# compute training error
	ypred = predict(model, data)
	loss_train = loss(model, ypred, labels)
	model.loss_train = loss_train

end

## predict

function predict(model::lda, data)

	w = model.weights
	b = model.intercept
	c = model.bias 

	y_pred = dropdims(sign.(data*w' .+ b .- c), dims = 2)
	y_pred = convert.( Int64, y_pred)

	i0_pred = findall(y_pred .== -1)
	i1_pred = findall(y_pred .== 1)

	y_pred[i0_pred] .= model.labels[1]
	y_pred[i1_pred] .= model.labels[2]

	return(y_pred)

end

## 0-1 loss for LDA

function loss(model::lda, ypred, y)

	acc = mean(ypred .== y)

end

## test function

function test(model::lda, offset = [1 1])

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
