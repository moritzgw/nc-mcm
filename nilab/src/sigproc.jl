
using FFTW, Statistics, DSP

export logbp, hann, car, stlogbp

"""
function hann(N::Int64)

	returns Hann window of length N

"""
function hann(N::Int64)

	n = 0:(N-1)
	w = 1/2 .* (1 .- cos.( (2*pi .* n) ./ (N-1) ) )

	return w

end

"""
function dft(x)

	computes and returns the Discrete Fourier Transform of x along the first dimension

	Note: This code is very slow! Use fft from FFTW for the Fast Fourier Transform

"""
function dft(x)

	N = size(x, 1)
	n = 0:(N-1)
	k = 0:(N-1)

	z = complex(zeros(size(x)))

	for i in k

		for j in n

			a = exp.( (-1im * 2*pi)/N .* n[j+1] .* k[i+1])

			z[i+1,:] += a .* x[j+1,:]

		end

	end

	return z

end

"""

function logbp(x, fs, freqs, window = hann())

"""
function logbp(x, fs, freqs, window = hann)

	# prepare
	sx = size(x)
	N = sx[1]
	xdims = cat(1, [i for i in sx[2:end]], dims = 1)
	xind = CartesianIndices(sx[2:end])

	powdims = deepcopy(xdims)
	powdims[1] = length(freqs) - 1
	pow = zeros(Tuple(powdims))

	# subtract mean
	mu = mean(x, dims = 1)
	x = x .- mu

	# compute DFT
	w = window(N)
	W = repeat(w, outer = xdims)	
	X = fft(x.*W, 1)

	# average across desired frequencies
	f = (0:N-1) ./ N .* fs

	for k = 2:length(freqs)

		mask_lower = f .>= freqs[k-1]
		mask_upper = f .< freqs[k]
		mask = mask_lower .* mask_upper
		scale = 2 / sum(w .^ 2) / fs
		pow[k-1,:,:] = 10 .* log10.(mean( (abs.(X[mask, xind]).^ 2) .* scale , dims = 1))

	end

	return pow

end

"""
Implements common average reference on eeg_file struct

"""
function car!(z::eeg_file)

	eeg = z.data

	 mu = mean(eeg, dims = 2)

	 for n in 1:size(eeg)[2]

	 	eeg[:,n] = eeg[:,n] - mu

	 end

	 z.data = eeg

	 return z

end

"""

Short-time Fourier transform log-bandpower estimation

Computes the short-time Fourier transform to estimate time-varying log-bandpower of signals.

Usage: X = stlogbp(x, fs, freqs, window_length = 1, window_step = 0.1, window = hann())

Input:
. x -- input signal [samples, channels]
. fs -- sampling frequency
. freqs -- vector of frequency bands to evaluate log-bandpower in
. windows_length -- length of each window for the STFT in seconds
. window_step -- step size for windows in seconds
. window = hann() -- windowing function to use

Output:
. X -- matrix of [log-bandpowers, channels, time points]
. t -- time indices (start of each window)
. f -- frequency indices (center of each window)

"""
function stlogbp(x, fs, freqs, window_length = 1, window_step = 0.1, winfunc = hann)

	# generate indices
	N = size(x)[1]
	xdims = ndims(x)

	wN = convert(Int, round(window_length * fs))
	stepN = convert(Int, round(window_step * fs))

	# compute log-bp in windows

	istart = 1
	iend = wN
	freqN = size(freqs)[1]-1

	# initialize arrrays (there must be a more elegant way to do this?)
	if xdims == 1

		X = zeros(freqN, 0)

	else

		X = zeros(freqN, size(x,2), 0)

	end

	t = zeros(0)

	while iend < N

		tmpX = logbp(x[istart:iend, :], fs, freqs, winfunc)
		X = cat(X, tmpX, dims = xdims+1)
		append!(t, istart/fs)

		istart = istart + stepN
		iend = iend + stepN

	end

	# collect output

	f = freqs[1:end-1] + diff(freqs) ./ 2

	return X, t, f

end