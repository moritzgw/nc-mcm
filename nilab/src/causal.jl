using Random
using Statistics

export markovian, simulate_markovian

"""
	
	markovian(z, K = 1000)

	expects a time-series of discrete states and returns 

	p  -- 	the p-value for the null-hypothesis that z is generated
			by a first-order Markov process.
	P  -- 	the first-oder transition probability matrix P(z[t]|z[t-1]) 

	where K is the number of of samples drawn from the null-distribution.

"""
function markovian(z, K = 1000)

	if eltype(z) == Any

		Pall = []
		statesall = []
		Mall = []
		Nall = []

		for zn in z

			P, states, M, N = compute_transition_matrix_lag2(zn, normalize = false)

			push!(Pall, P)
			push!(statesall, states)
			push!(Mall, M)
			push!(Nall, N)

		end

		@assert sum([q == statesall[1] for q in statesall]) == length(z) # check that states are identical across data sets

		P = reduce( (x,y) -> cat(x, y, dims = 4), Pall)
		M = sum(reduce(vcat, Mall))
		P = mean(P, dims = 4)[:,:,:,1] / M

	else

		P, states, M, N = compute_transition_matrix_lag2(z)	

	end

	# P(z[t]|z[t-1]) = P(z[t],z[t-1]) / P(z[t-1])
	Pz0z1 = sum(P, dims = 3)
	Pz1 = sum(P, dims = [1, 3])
	P1 = Pz0z1 ./ Pz1

	# P(z[t]|z[t-1],z[t-2]) = P(z[t],z[t-1],z[t-2]) / P(z[t-1],z[t-2])
	Pz1z2 = sum(P, dims = 1)
	P2 = P ./ repeat(Pz1z2, outer = (N,1,1))
	P2[isnan.(P2)] .= 0

	# statistical test
	# K = 1000 # number of samples from H0
	TH0 = zeros(K)

	for kperm = 1:K

		zH0, _ = simulate_markovian(M, P1)

		PH0 = zeros(N,N,N)

		for m = 3:M

			i = zH0[m]
			j = zH0[m-1]
			k = zH0[m-2]

			PH0[i,j,k] = PH0[i,j,k] .+ 1

		end

		PH0 = PH0 ./ (M-2)

		Pz1z2H0 = sum(PH0, dims = 1)
		P2H0 = PH0 ./ repeat(Pz1z2H0, outer = (N,1,1))
		P2H0[isnan.(P2H0)] .= 0

		TH0[kperm] = sum(var(P2H0, dims = 3))

	end

	# compute p-value
	T = sum(var(P2, dims = 3))
	p = 1 - mean(T .>= TH0)

	# return results
	P_return = permutedims( dropdims(P1, dims = 3))

	return p, P_return

end

"""
	simulate_markovian(M, P = [], N = [])

	M -- number of samples
	P -- transition probability matrix
		 P_{i,j}  = P(x = i | x = j)
	N -- dims for random transition probability matrix

	Initial distribution is assumed to be uniform across states.

"""
function simulate_markovian(M, P = [], N = [])

	# create random transition matrix

	if isempty(P)

		P = rand(N,N)
		P = P ./ repeat(sum(P, dims = 1), outer = (N,1))
		# P = [0.99 0 0.01; 0.01 0.99 0 ; 0 0.01 0.99]

	else

		N = size(P)[1]

	end

	CP  = cumsum(P, dims = 1)

	# generate data
	z = zeros(Int64, M)
	z[1] = randperm(N)[1]

	for m = 2:M

		z[m] = findfirst( CP[:,z[m-1]] .>= rand(1) )

	end

	return z, P

end


"""
	compute_transition_matrix_lag2(z)

	Computes the state transition matrix T of time series z with two lags and returns

	T -- the transition matrix
	states -- the unique elements of z
	M -- the number of unique elements of z
	N -- the length of z

"""
function compute_transition_matrix_lag2(z; normalize = true)

	states = sort(unique(z))
	M = length(z)
	N = length(states)

	# rename state
	x = zeros(Int64, size(z))

	for (i, state) in enumerate(states)

		j = findall(z .== state)
		x[j] .= i

	end

	# P(z[t],z[t-1],z[t-2])
	P = zeros(N,N,N)

	for m = 3:M

		i = x[m]
		j = x[m-1]
		k = x[m-2]

		P[i,j,k] = P[i,j,k] .+ 1

	end

	if normalize

		P = P ./ (M-2)

	end

	return P, states, M, N

end
