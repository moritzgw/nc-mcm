
using Random
using Statistics

export permtest, confusionmatrix, panova, fvalue

"""

	function permtest(f::Function, x, y, nperm = 10000, two_sided = true)

		f: f(x,y) computes one-dimensional test-statistic.

		Samples are assumed to be in the first dimension of x and y.
		Permutations are done on x.

	Returns (test_statistic, p)

"""
function permtest(f::Function, x, y, nperm = 10000, two_sided = true)

	# Check that array have same number of samples

	mx = size(x)[1]
	my = size(y)[1]

	if mx !== my 

		error("Unequal number of samples for x and y!")

	end

	# Original test statistic

	test_stat = f(x, y)

	# Sample from H0	

	H0 = zeros(nperm)
	sx = size(x)

	for n = 1:nperm

		# permute
		x2dim = reshape(x, mx, :)
		iperm = randperm(mx)
		x2dim = x2dim[iperm, :]
		xH0 = reshape(x2dim, sx)

		# sample from H0
		H0[n] = f(xH0, y)

	end

	if two_sided == true

		p = mean(abs.(H0) .>= abs(test_stat)) + 1/nperm

	else

		p = mean(H0 .>= test_stat) + 1/nperm

	end		

	return test_stat, p, H0

end

"""

	function pairedpermtest(f::Function, x, y, nperm = 10000, two_sided = true)

		f: f(x,y) computes one-dimensional test-statistic.

		Samples are assumed to be in the first dimension of x and y.

	Returns (test_statistic, p)

"""
function pairedpermtest(f::Function, x, y, nperm = 10000, two_sided = true)

	# Check that array have same number of samples

	mx = size(x)[1]
	my = size(y)[1]

	if mx !== my 

		error("Unequal number of samples for x and y!")

	end

	# Original test statistic

	test_stat = f(x,y)

	# Sample from H0	

	H0 = zeros(nperm)
	sx = size(x)

	for n = 1:nperm

		# permute
		x2 = reshape(x, mx, :)
		y2 = reshape(y, my, :)
		z = cat(x, y, dims = 2)
		iperm1 = rand(1:2, mx)
		iperm2 = convert.(Int64, 1 ./ (iperm1 ./ 2))
		permmask = cat(iperm1, iperm2, dims = 2)
		
		# sample from H0
		zH0 = z[permmask]		
		H0[n] = f(zH0[:,1], zH0[:,2])

	end

	if two_sided == true

		p = mean(abs.(H0) .>= abs(test_stat)) + 1/nperm

	else

		p = mean(H0 .>= test_stat) + 1/nperm

	end		

	return test_stat, p

end

"""
	confusionmatrix(ypred, y)

	Returns C, labels
	with 'C' the confusion matrix (raw count, true labels in rows).

"""
function confusionmatrix(ypred, y)

	labels = unique(y)
	K = length(labels)

	C = zeros(K, K)

	for (i, true_label) in enumerate(labels)

		for (j, pred_label) in enumerate(labels)

			itrue = findall(y .== true_label)
			
			C[i,j] = sum(ypred[itrue] .== pred_label)

		end

	end

	return C, labels

end


"""

	stdz(z)

	standardizes the 1D variable z (outlier rejection not yet incorporated)

"""
function stdz(z)

	z = z .- mean(z)
	z = z ./ std(z)

	return z

end


"""
	
	cumprob(z)

	Returns cumulative probability distribution (x, P(z <= x))


"""
function cumprob(z)

	N = length(z)
	zs = sort(z)
	P = 1:N

	return zs, P

end


"""

	fvalue(x, c)

	Computes and returns variance ratio before and after subtracting level means from 1D data in x of size [N samples]
	with discrete levels given in c [N samples].

"""
function fvalue(x, c)

	N = size(x)[1]
	levels = unique(c)
	K = length(levels)

	indices = []
	level_means = zeros(K)	

	for k = 1:K

		i = nifind(c, levels[k])
		push!(indices, i)

		level_means[k] = mean(x[i])

	end

	xm = zeros(size(x))

	for n = 1:N

		xtmp = x[n] .- level_means[nifind(levels, c[n])]
		xm[n] = xtmp[1]

	end

	F = var(xm) / var(x)

	return F

end


""" 

	panova(x, c, nperm = 10000)

	Computes and returns variance ratio (before / after mean substraction) and p-value of ANOVA for
	1D-data x and discrete levels in c (both N samples).

"""
function panova(x, c, nperm = 10000)

	g(b, a) = fvalue(a, b)

	F, ip = permtest(g, c, x, nperm, false)

	p = 1 - ip

	return F, p

end