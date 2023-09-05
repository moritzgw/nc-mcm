module nilab

	# println("Precompiling nilab.")

	function __init__()
	    println("Loading nilab.")
	end

	using Statistics
	using Random
	using Flux
	using Combinatorics

	## supervised models
	include("supervised_models/lda_model.jl")
	include("supervised_models/linreg_model.jl")
	include("supervised_models/logreg_model.jl")

	## statistical tests
	include("nistats.jl")

	# loading data
	include("load.jl")

	# unsupervised models
	include("unsupervised_models.jl")

	# causal reasoning and inference
	include("causal.jl")

	# signal processing
	include("sigproc.jl")

	# plotting
	include("plot.jl")

	# auxiliary functions
	include("niaux.jl")

	# # interpretable machine learning
	# include("iml.jl")

	# multilevel causal modeling
	include("mcm.jl")

	## export functions defined in base
	
	export splitdata, crossvalidate, learningcurve, sgd, testdata, multiclasstrain, multiclasspredict, csgd

	## function definitions in based

	"""
		splitdata(m, split, random = true)

		Input:

			m: number of samples
			split:
				if integer: number of equally sized splits for cross-validation
				if float in [0,1]: percentage of test data for split into training / test
			random:
				if true: Shuffle data indices before splitting
				if false: Consecutive split in sequence train, test

		Returns: 

				if split is integer: Matrix of indices of size [folds, samples]
				if split is float:	arrays itrain, itest

	"""
	function splitdata(m, split, random_splits = true)


		if random_splits == true

			samples = randperm(m)

		else

			samples = 1:m

		end

		if typeof(split) == Int

			samples = reshape(samples, (:, split))'

			return samples 

		else

			nsplit = Int64(round(m * (1 - split)))
			itrain = samples[1:nsplit]
			itest = samples[(nsplit+1):end]

			return itrain, itest

		end

	end

	"""
		function crossvalidate(model, data, labels; nrsplits = 10, random_splits = true, H0 = false)

		returns (train_loss, test_loss) across folds 

		If H0 = true, labels in the training set are permuted before training.

	"""
	function crossvalidate(model, data, labels; nrsplits = 10, random_splits = true, H0 = false)

		# run cv
		m = size(data)[1]
		folds = splitdata(m, nrsplits, random_splits)

		train_loss = []
		test_loss = []

		for i in 1:nrsplits

			train_mask = trues(size(folds))
			train_mask[i,:] .= false
			test_mask = falses(size(folds))
			test_mask[i,:] .= true

			curmodel = deepcopy(model)

			# determine whether to run binary- or multiclass classification
			if length(unique(labels)) > 2

				ytrain = labels[folds[train_mask]]

				if H0 == true 

					shuffle!(ytrain)

				end

				trained_models = multiclasstrain(curmodel, data[folds[train_mask], :], ytrain)
				ypred_train = multiclasspredict(trained_models, data[folds[train_mask], :])
				ypred_test = multiclasspredict(trained_models, data[folds[test_mask], :])
				loss_train = loss(curmodel, ypred_train, labels[folds[train_mask]])
				loss_test = loss(curmodel, ypred_test, labels[folds[test_mask]])

			else

				ytrain = labels[folds[train_mask]]

				if H0 == true 

					shuffle!(ytrain)

				end

				train!(curmodel, data[folds[train_mask], :], ytrain)
				ypred_test = predict(curmodel, data[folds[test_mask], :])
				ypred_train = predict(curmodel, data[folds[train_mask], :])
				loss_train = loss(curmodel, ypred_train, labels[folds[train_mask]])
				loss_test = loss(curmodel, ypred_test, labels[folds[test_mask]])

			end

			push!(train_loss, loss_train)
			push!(test_loss, loss_test)

		end

		return train_loss, test_loss

	end

	"""
		function learningcurve(model, data, labels, steps = 10, reps = 10)

			Draws random subsets of training data and predicts on remaining test data.

			Returns learningcurve_train, learningcurve_test @ [train_size, repetitions]

	"""
	function learningcurve(model, data, labels, steps = 10, reps = 10, random_splits = true)

		m = size(data)[1]
		lc_train = zeros(steps-1, reps)
		lc_test = zeros(steps-1, reps)

		for istep = 1:(steps-1)

			 train_frac = istep / steps 

			for irep = 1:reps

				curmodel = deepcopy(model)

				itrain, itest = splitdata(m, 1 - train_frac, random_splits)
				train!(curmodel, data[itrain, :], labels[itrain])
				ypred = predict(curmodel, data[itest, :])

				lc_train[istep, irep] = curmodel.loss_train
				lc_test[istep, irep] = loss(curmodel, ypred, labels[itest])

			end

		end

		return lc_train, lc_test

	end

	"""
		function sgd(f, x0, eta0 = 0.1, tol = 1e-6, max_steps = 1000) implements stochastic gradient descent

			Input:

				f: loss function to minimize
				x0: initial parameter (vector) of f
				eta0: Initial learning rate (reduced by 1/sqrt(steps))
				max_steps: Maximum number of iterations
				tol: relative change in loss function for convergence

			Returns: Paramester estimate

	"""
	function sgd(f, x0, eta0 = 0.1, tol = 1e-6, max_steps = 1e5)

		# initialize
		t = 1
		dtol = 1
		eta = eta0
		x = tuple(x0)
		df(x) = Flux.gradient(f, x)

		# optimization loop
		while (t <= max_steps) & (dtol > tol)

			# update
			f0 = f(x[1])
			eta = eta0 / sqrt(t)
			x = x .- eta .* df(x[1])
			f1 = f(x[1])

			# flow control
			t = t + 1
			dtol = abs(f1-f0)

			println("Objective: ", f1)

		end

		println("Terminated after ", t-1, " of ", max_steps, " steps at tolerance ", dtol, " with initial learning rate ", eta0, ".")

		return x

	end


	"""
		function csgd(f, x0, eta0 = 0.1, tol = 1e-6, max_steps = 1000) implements linearly-constrained stochastic gradient descent

			Input:

				f: loss function to minimize
				g: linear constraint to satisy (g(x) = 0)
				x0: initial parameter (vector) of f
				eta0: Initial learning rate (reduced by 1/sqrt(steps))
				max_steps: Maximum number of iterations
				tol: relative change in loss function for convergence

			Returns: Paramester estimate

	"""
	function csgd(f, g, x0; eta0 = 0.1, tol = 1e-6, max_steps = 1e5)

		println("WARNING: CODE UNDER DEVELOPMENT, DOES NOT WORK YET! CONSTRAINT IS KEPT CONSTANT RELATIVE TO INITIAL CONDITION.")

		# initialize
		t = 1
		dtol = 1
		eta = eta0
		x = x0
		ddf(x) = Flux.gradient(f, x)
		ddg(x) = Flux.gradient(g, x)

		eta_traj = [eta]
		x_traj = [convert.(Float64, x)]

		# # project initial x to constraint
		# dfx = ddf(x)[1]
		# dgx = ddg(x)[1]
		# x0proj = x0 - x0'*dgx / (dgx'*dgx) * dgx
		# x = x0proj
		# push!(x_traj, x)

		# optimization loop
		while (t <= max_steps) & (dtol > tol)

			# project
			dfx = ddf(x)[1]
			dgx = ddg(x)[1]
			dxproj = dfx - dfx'*dgx / (dgx'*dgx) * dgx

			# update
			f0 = f(x)
			g0 = g(x)
			eta = eta0 / sqrt(t)
			xnew = x .- eta .* dxproj
			f1 = f(xnew)
			g1 = g(xnew)
			x = xnew

			# flow control
			t = t + 1
			dtol = abs(f1-f0)

			println("Objective / constraint: ", f1, " / ", g1)

			push!(eta_traj, eta)
			push!(x_traj, x)

		end

		println("Terminated after ", t-1, " of ", max_steps, " steps at tolerance ", dtol, " with initial learning rate ", eta0, ".")

		return x#, x_traj, eta_traj

	end


	function testdata(offset = [1 1], m = 100, sigma = 2.5)

		# set parameters
		N = length(offset)
	
		# create data
		m2 = convert(Int64, m/2)
		labels = cat(zeros(m2), ones(m2), dims = 1);
		data0 = sigma .* randn(m2, N) + repeat(offset, outer = [m2, 1]);
		data1 = sigma .* randn(m2, N) - repeat(offset, outer = [m2, 1]);
		data = cat(data0, data1, dims = 1)

		return data, labels

	end

	mutable struct multiclass_model

		labels
		label_pairs
		models

		multiclass_model() = new()

	end

	"""
		models = multiclasstrain(model, data, labels)

		Implements multiclass training using pairwise binary training.

	"""
	function multiclasstrain(model, data, labels)

		# create classifier pairs
		label_names = unique(labels)
		pairs = collect(combinations( label_names, 2))
		K = length(pairs)
		binary_models = []

		for k = 1:K

			# copy original model
			cur_model = deepcopy(model)

			# get data for binary training
			isamples = [i for i in 1:length(labels) if labels[i] in pairs[k]]
			binary_data = selectdim(data, 1, isamples)
			binary_labels = labels[isamples]

			# run brinary training
			train!(cur_model, binary_data, binary_labels)

			# store model
			push!(binary_models, cur_model)

		end

		final_model = multiclass_model()
		final_model.labels = label_names
		final_model.label_pairs = pairs
		final_model.models = binary_models

		return final_model

	end

	"""
		predictions = multiclasspredict(model, data)

		Implements multiclass prediction for models trained with multiclasstraining().

	"""
	function multiclasspredict(models, data)

		# initialize
		M = size(data)[1]
		K = length(models.label_pairs)
		L = length(models.labels)

		# predict
		Y = zeros(M,K)

		for k = 1:K

			Y[:,k] = predict(models.models[k], data)

		end

		# pick majority vote
		Z = zeros(M,L)

		for l = 1:L

			Z[:,l] = sum( Y .== models.labels[l], dims = 2)

		end 

		ipred = argmax(Z, dims = 2)

		ypred = [models.labels[ipred[i][2]] for i in 1:M]

		return ypred

	end

end

