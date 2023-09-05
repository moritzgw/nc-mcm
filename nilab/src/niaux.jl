export nifind, nihist

function nifind(z, x)

	ind = [m for m in 1:length(z) if z[m] == x]

	return ind
end

function nihist(z; n = 100)

	# create bins
	zrange = [minimum(z), maximum(z)]
	dz = (zrange[2] - zrange[1]) / n
	bins = zrange[1]:dz:(zrange[2])

	h = zeros(n)

	# fill bins
	for k = 1:length(bins)-1

		h[k] = sum((z .>= bins[k]) .& (z .< bins[k+1]))

	end

	return h

end