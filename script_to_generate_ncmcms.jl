###########
# Imports #
###########

using nilab
using CSV
using DataFrames
using PyCall
using LinearAlgebra
using Statistics
using Gnuplot
using JLD2

#############
# Functions #
#############

function loadce(file, set; delbehavneurons = true) # loads the raw data and behavioral labels

	vars = loadmat(file)
	ikey = file[1:end-4]
	rawdata = vars[ikey]

	# neuronal data
	x = rawdata["deltaFOverF_bc"][set]

	# neurons
	neurons = rawdata["NeuronNames"][set]

	# HACK TO FIX THAT ONE NEURON'S NAME IN WORM 4 IS NOT PROPERLY STORED
	if set == 4

		neurons[end] = neurons[end][1]
	end

	# exclude neuron used for behavioral labeling
	behav_neurons = ["AVAL", "AVAR", "SMDDR", "SMDDL", "SMDVR", "SMDVL", "RIBR", "RIBL"]

	if delbehavneurons == true

		delx = []

		for neuron in behav_neurons

			idel = findall(neuron .== neurons)

			if isempty(idel)

				println("Neuron ", neuron, " not found in data set ", set)

			else

				push!(delx, idel[1][2])

			end

		end

		xdelmask = trues(size(x)[2])
		xdelmask[delx] .= false
		x = x[:, xdelmask]

	end

	# states
	ydict = rawdata["States"][set] 
	K = collect(keys(ydict))
	M = size(x,1)
	yfull = zeros(M, length(K))

	for (i, k) in enumerate(K)

		yfull[:, i] = ydict[k]

	end

	y = [argmax(yfull[i,:]) for i in 1:M]

	return x, y, K, neurons

end

	
#################
# Learn NC-MCMs #
#################

mcm_models = []

for iworm = 1:5

	# Load data of one worm
	x, b, blabels, neurons = loadce("NoStim_Data.mat", iworm)

	# Compute cross-validated classification accuracy
	cv_folds = 10
	base_model = logreg()
	M = size(x)[1] # number of samples 
	Mcv = mod(M, cv_folds)
	isamples = 1:(M-Mcv) # ensure same number of samples in each fold 

	trainloss, testloss = crossvalidate(base_model, x[isamples,:], b[isamples], nrsplits = cv_folds)
	println(mean(testloss))

	# Learn MCM
	mcm_model = learn_mcm(x, b)
	push!(mcm_models, mcm_model)

	# Store in JLD2
	@save string("nc_mcm_model_worm_", iworm, ".jld2") mcm_model

	# Store for plotting
	np = pyimport("numpy")
	np.savez(string("nc_mcm_model_worm_", iworm), mcm_model.x, mcm_model.c, mcm_model.b, blabels, mcm_model.p, mcm_model.bprob)

end


######################################################################
# Extract and store raw data with all neurons for plotting in Python #
######################################################################

# neurons
identified_neurons_numbers = []
identified_neurons_names = []

for iworm = 1:5

	# raw data
	x, b, blabels, neurons = loadce("NoStim_Data.mat", iworm, delbehavneurons = false)
	np.savez(string("raw_full_data_worm_", iworm), x)

	nr_neurons = length(neurons)
	lettersum = zeros(Int64, nr_neurons)

	for neuron in 1:nr_neurons

		println(neuron)
		lettersum[neuron] = sum([isletter(i) for i in neurons[neuron]])

	end

	ineurons = findall(lettersum .> 0)
	push!(identified_neurons_numbers, ineurons .- 1) # saving for Python!
	push!(identified_neurons_names, neurons[ineurons])

end

np.savez("known_neurons", identified_neurons_numbers, identified_neurons_names)

