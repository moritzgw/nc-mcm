
using Random
using Statistics

export BLA, train!, predict, loss, test

## model struct

mutable struct BLA

	# model parameters
	labels 		# names of binary labels
	N::Int64 	# number of features
	m::Int64 	# number of samples
	weights
	intercept
	bias
	loss_train

	BLA() = new()

end

## train

function train!(model::BLA, data, labels)

	# nr samples
	m = size(data)[1]

	# train model

	# store parameters

	# compute training error
	ypred = predict(model, data)
	loss_train = loss(model, ypred, labels)
	model.loss_train = loss_train

end

## predict

function predict(model::BLA, data)


	return(y_pred)

end

## loss for BLA

function loss(model::BLA, ypre, y)


end

## test function

function test(model::BLA)

	# set parameters

	# create data

	# train model and predict
	train!(model, data, labels)
	y_pred = predict(model, data)
	acc = loss(model, y_pred, labels)
	println("Train: ", acc) 

	return(model)

end
