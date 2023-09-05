#%% Module implementing neuro-cognitive multilevel causal models (NC-MCMs) 

using Statistics
using Clustering

export learn_mcm, kmeans_mcm, kmedoids_mcm

# Define MCM model class
mutable struct mcm

	# model parameters
	x 		# raw data
	K 		# number of cognitive states
	c 		# cognitive states
	p 		# p-values for rejecting H0: Markov chain for each k in 1:K
	b 		# behavioral labels
	bpred 	# predicted behavioral labels
	bprob	# predicted behavioral probabilities

	# initial values
	mcm() = (z = new();
		x = [];
		K = 0;
		c = [];
		p = [];
		b = [];
		bpred = [];
		bprob = [];
		return z )

end


#%% 

"""
	This function learns a multi-level causal model from micro-level states x and behaviors y:

	function mcm(x, b ; dimreduction = [], predmodel = logreg, clustering = kmeans, markov_test = markovian)

	Input:

		x 	neuronal data [smaples x features]
		b 	discrete behavioral labels [samples]

	It relies on the following processing steps:

		1. Applies dimensionality reduction method `dimreduction` to input data x [none implemented as of now]
		2. Uses prediction model `predmodel` to predict the probabilities of each behavior at every sample of x
		3. Clusters the predicted probabilities using the `clustering` method to generate macroscopcic states
		4. Uses the test given by `markov_test` to test H0: th macroscopic state transitions form a Markov chain

	It returns a mcm model with the following elements:

		x 		raw data
		K 		number of cognitive states
		c 		cognitive states
		p 		p-values for rejecting H0: Markov chain for each k in 1:K
		b 		behavioral labels
		bpred 	predicted behavioral labels
		bprob	predicted behavioral probabilities


"""
function learn_mcm(x, b ; dimreduction = [], predmodel = logreg, clustering = kmeans_mcm, markov_test = markovian)

	# Dimensionality reduction

	# Prediction
	model = predmodel()

	Nb = length(unique(b))
	M, D = size(x)

	if Nb == 2

		println("Training 2-class prediction model...")
		train!(model, x, b)
		bpred = predict( model, x)
		bprob = predictprob(model, x)

	else

		println("Training ", Nb, "-class prediction model...")
		model = multiclasstrain(model, x, b)
		bpred = multiclasspredict( model, x)

		K = length(model.models)
		bprob = zeros(M, K)

		for k = 1:K

			bprob[:, k] = predictprob(model.models[k], x)

		end

	end

	acc = mean(bpred .== b)
	println("Prediction accuracy (not cross-validated): ", acc)

	# Clustering
	println("Running clustering...")
	c, p = clustering(bprob)

	# Test for Markovianity
	println("Testing for Markovianity...")
	K = size(c)[2]
	p = zeros(K)

	for k in 1:K

		p[k], _ = markov_test(c[:,k])

	end

	# Store results
	mcm_model = mcm()
	mcm_model.x = x
	mcm_model.K = K
	mcm_model.c = c
	mcm_model.p = p
	mcm_model.b = b
	mcm_model.bpred = bpred
	mcm_model.bprob = bprob

	return mcm_model

	println("Done.")

end


function kmeans_mcm(z ; repetitions = 100, max_clusters = 20)

	println("Clustering with k-means...")

	M, K = size(z)
	p_markov = zeros(max_clusters, repetitions)
	call = zeros(M, max_clusters, repetitions)

	for reps = 1:repetitions

		println("Running ", reps, " of ", repetitions, " repetitions.")

		for nrclusters = 1:max_clusters

			# k-means
			clusters = kmeans( z', nrclusters) # ; init=:rand)
			ctmp = assignments(clusters)
			
			p, _ = markovian(ctmp)

			p_markov[nrclusters, reps] = p
			call[:, nrclusters, reps] = ctmp

		end

	end

	# for each number of clusters, return p-value and cluster assignments with maximal p-value
	i = argmax(p_markov, dims = 2)
	p = dropdims( p_markov[i], dims = 2)
	c = dropdims(call[:, i], dims = 3)

	return c, p

end


function kmedoids_mcm(z ; repetitions = 10, max_clusters = 10)

	println("Clustering with k-medoids...")

	M, K = size(z)
	p_markov = zeros(max_clusters, repetitions)
	call = zeros(M, max_clusters, repetitions)

	# Compute distance matrix

	println("Computing distance matrix...")

	D = zeros(M, M)

	for n1 = 1:M

		for n2 = 1:M

			D[n1, n2] = norm(z[n1, :] - z[n2, :])

		end

	end

	# Carry out clustering
	for reps = 1:repetitions

		println("Running ", reps, " of ", repetitions, " repetitions.")

		for nrclusters = 1:max_clusters

			# k-means
			clusters = kmedoids( D, nrclusters) # ; init=:rand)
			ctmp = assignments(clusters)
			
			p, _ = markovian(ctmp)

			p_markov[nrclusters, reps] = p
			call[:, nrclusters, reps] = ctmp

		end

	end

	# for each number of clusters, return p-value and cluster assignments with maximal p-value
	i = argmax(p_markov, dims = 2)
	p = dropdims( p_markov[i], dims = 2)
	c = dropdims(call[:, i], dims = 3)

	return c, p

end