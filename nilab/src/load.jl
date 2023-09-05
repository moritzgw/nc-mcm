using MAT
using PyCall

export loadmat, loadeeg, info, loadopenbci, loadnp

function loadmat(file)

	vars = matread(file)

end

mutable struct eeg_file

	# header information
	filename
	fs
	nr_chans
	chan_name
	chan_reference
	chan_resolution
	chan_unit

	# eeg data
	data

	# marker information
	marker_type
	marker_description
	marker_position
	marker_length
	marker_channel_number

	# initial values
	eeg_file() = (z = new();
		return z )

end

"""
	function loadeeg(file)

	Function for loading BrainVision .vhdr files.

	Input:

		file:	filename without extension(s)

	Returns:

		structure eeg_file

"""
function loadeeg(file)

	# create struct
	dataset = eeg_file()
	dataset.filename = file

	# set file names
	header_file_name = string(file, ".vhdr")
	eeg_file_name = string(file, ".eeg")
	marker_file_name = string(file, ".vmrk")

	# read header file
	header = readlines(header_file_name)

	# data format checks
	if !occursin("Brain Vision Data Exchange Header File Version 1.0", header[1])

		error("Not a Brain Vision 1.0 file!")

	end

	irow_data_format = findall(occursin.("BinaryFormat", header))[1]
	data_format = split(header[irow_data_format], "=")[end]

	if data_format != "IEEE_FLOAT_32"

		error("EEG data must be in IEEE_FLOAT_32 encoding.")

	end

	# extract values
	dataset.nr_chans = getvhdrkey(header, "NumberOfChannels", Int)
	dataset.fs = 1 / (getvhdrkey(header, "SamplingInterval", Float64) * 1e-6)

	chan_name = []
	chan_reference = []
	chan_resolution = []
	chan_unit = []

	row_chans = findall(occursin.("[Channel Infos]", header))[1] + 1

	for ichan in row_chans:(row_chans + dataset.nr_chans - 1)

		chan_entries = split(header[ichan], ('=', ','))

		push!(chan_name, chan_entries[2])
		push!(chan_reference, chan_entries[3])
		push!(chan_resolution, parse(Float64, chan_entries[4]))
		push!(chan_unit, chan_entries[5])

	end

	dataset.chan_name = chan_name
	dataset.chan_reference = chan_reference
	dataset.chan_resolution = chan_resolution
	dataset.chan_unit = chan_unit

	# read eeg file
	eeg = read(eeg_file_name)
	eeg = reshape( eeg, 4, :)
	eeg = reinterpret(Float32, eeg)
	eeg = reshape(eeg, dataset.nr_chans, :)
	eeg = dataset.chan_resolution .* eeg

	dataset.data = eeg'

	# read marker file
	marker = readlines(marker_file_name)

	marker_type = []
	marker_description = []
	marker_position = []
	marker_length = []
	marker_channel_number = []

	row_marker = findall(occursin.("[Marker infos]", marker))[1] + 1

	for imarker = row_marker:size(marker)[1]

		marker_entries = split(marker[imarker], ('=', ','))

		push!(marker_type, marker_entries[2])
		push!(marker_description, marker_entries[3])
		push!(marker_position, parse(Int32, marker_entries[4]))
		push!(marker_length, parse(Int32, marker_entries[5]))
		push!(marker_channel_number, parse(Int32, marker_entries[6]))

	end

	dataset.marker_type = marker_type
	dataset.marker_description = marker_description
	dataset.marker_position = marker_position
	dataset.marker_length = marker_length
	dataset.marker_channel_number = marker_channel_number

	# return 
	return dataset

end

function getvhdrkey(header, key, key_type)

	irow = findall(occursin.(key, header))[1]
	value = parse(key_type, split(header[irow], "=")[end])

end

"""
	function summary(z::eeg_file)

	Displays summary of struct EEG file

"""
function info(z::nilab.eeg_file)

	println("Filename: ", z.filename)
	println("Sampling rate: ", z.fs, " Hz")
	println("Number of channels: ", z.nr_chans)
	println("Length: ", size(z.data)[1] / z.fs, " s")
	
	if isdefined(z, :marker_type)

		println("Markers types: ", string(unique(z.marker_type)))
		
		markers = unique(z.marker_type)
		nrmarkers = Array{Int32}([])

		for m in markers

			push!(nrmarkers, sum(z.marker_type .== m))

		end

		println("Marker types #: ", string.(nrmarkers))

		println("Markers descriptions: ", string(unique(z.marker_description)))
		
		markers = unique(z.marker_description)
		nrmarkers_desc = Array{Int32}([])

		for m in markers

			push!(nrmarkers_desc, sum(z.marker_description .== m))

		end

		println("Marker description #: ", string.(nrmarkers_desc))

	end

end

"""
	
	function loadopenbci(file)

	Returns eeg_file struct

	Note:
	* Currently no marker information read-out incorporated.
	* Reading based on BrainWaveBank format!
	* CLK_ADJUST not taken care of -- what is that anyhow?

"""
function loadopenbci(file)

	# read raw data
	raw = readlines(file)

	# identify header and data
	rows_header = findall([z[1] for z in raw] .== '%')
	rows_data = [i for i in 1:size(raw)[1] if !(i in rows_header)]

	# extract data set properties
	nrchans = length(split(raw[rows_data[1]], ',')) - 2
	samples = length(rows_data)

	row_fs = findfirst(occursin.("Sample Rate", raw))
	fs = parse(Float32, split(raw[row_fs], ' ')[end-1])

	# parse EEG data
	data = zeros(Float32, samples, nrchans)
	time = zeros(Int64, samples)

	for n in 1:samples

		row = raw[rows_data[n]]

		row_split = split(row, ',')
		row_data = row_split[2:end-1]
		row_time = row_split[end]

		if length(row_data) == nrchans

			time[n] = parse(Int64, row_time)
			data[n,:] = parse.(Float32, row_data)

		else

			println("Warning: Row ", n, " does not have the correct number of channels, skipping.")

		end

	end

	# write to struct
	eeg = eeg_file()

	eeg.filename = file
	eeg.fs = fs
	eeg.nr_chans = nrchans
	eeg.chan_unit = "muV"
	eeg.data = data

	return eeg

end

"""
	
	function loadnp(file)

	Reads Python NumPy file via PyCall.

"""
function loadnp(file)

	np = pyimport("numpy")

	z = np.load(file)

	return z

end