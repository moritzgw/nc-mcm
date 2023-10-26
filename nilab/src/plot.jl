# using Gaston

# export plotbp #, closeall
export gnu_generate_tic_string

# """

# 	function plotbp(freqs, Z)

# """
# function plotbp(freqs, Z)

# 	fig = figure()

# 	display(plot(freqs[1:end-1] .+ (freqs[2] - freqs[1])/2, Z[:,1], w = :l, Axes(grid = :on)))
	
# 	for k in 2:size(Z,2)
		
# 		display(plot!(freqs[1:end-1] .+ (freqs[2] - freqs[1])/2, Z[:,k], w = :l))

# 	end

# end

# """

# function closeall()

# Closes all Gaston / GNU Plot figures

# """
# function closeall()

# 	nf = figure()

# 	for n = 1:nf

# 		closefigure(n)

# 	end

# end

function gnu_generate_tic_string(labels, pos)

	if isempty(pos)

		pos = 0:(lengt(labels) - 1)

	end

	out = string()

	for k in 1:length(labels)

		out = string(out, string("'", labels[k], "' ", pos[k], ","))

	end

	return out

end
