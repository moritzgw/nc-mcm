def plot_trajectory(file, file2, nc = 3, plottype = 'cognitive', prob = False):

	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	from scipy.signal import butter, filtfilt

	data = np.load(file)
	x = data["arr_0"] 
	c = data["arr_1"][:, nc-1]
	b = data["arr_2"]-1
	labels = data["arr_3"]
	p_markov = data["arr_4"]
	bprob = data["arr_5"]

	if file2 != []:

		data2 = np.load(file2)
		x = data2["arr_0"] 
		b = data2["arr_2"]-1
		labels = data["arr_3"]
		c = c[15:] # shorten data for bundle-net

	print("p-value: " + str(p_markov[nc-1]))

	if plottype == 'cognitive':

		y = c

	elif plottype == 'behavior':

		y = b

	else:

		error('plottype not recognized')

	if prob == True:

		x = bprob
		X = np.corrcoef(x.T)
		d, V = np.linalg.eig(X)
		x = np.dot(x, V[:, 0:3])

	if x.shape[1] > 3:

		print("Reducing x to 3 dims by PCA...")

		X = np.corrcoef(x.T)
		d, V = np.linalg.eig(X)
		x = np.dot(x, V[:, 0:3])

	Y = np.unique(y)

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	colors = ['b','g','r','c','m','gray','y','peru']

	for i in range(len(Y)):

		iy = np.where(y == Y[i])[0]

		# curax.arrow(0, 0, 0, -2.5, color = 'w', width=0, linewidth=0) # force offset to avoid overlap of behavioral legend with data
		for j in iy[0:-1]:

			dx = 0.3*(x[j+1, 0] - x[j, 0])
			dy = 0.3*(x[j+1, 1] - x[j, 1])
			dz = 0.3*(x[j+1, 2] - x[j, 2])
			# ax.arrow(x[j, 0], x[j, 1], dx, dy, width=0.01, color = colors[int(i)], linewidth=0.01)
			ax.quiver(x[j, 0], x[j, 1], x[j, 2], dx, dy, dz, linewidth = 1, length=2, normalize=False, color = colors[int(i)], arrow_length_ratio=0.5)

		# ax.set_axis_off()

	legend = [Line2D([0], [0], color=colors[n], lw=2) for n in range(len(Y))]
	
	if plottype == 'cognitive':

		ax.legend(legend, ["C"+str(int(z)) for z in Y])

	elif plottype == 'behavior':

		ax.legend(legend, [z for z in labels])

	else:

		error('plottype not recognized')



def plot_ncmcmgraph(file, nset = 3, max_width = 1000, max_size = 1000, size_arrows = 10, textsize = 12, T_thresh = 0, save = False, reorder = [], Nsmooth = [], separate_plot = True, bspread = 0.25):

	#############
	#%% Imports #
	#############

	import networkx as nx
	import matplotlib.pyplot as plt
	import matplotlib
	import numpy as np
	from scipy.stats import mode

	###############
	#%% Load data #
	###############

	data = np.load(file)
	x = data["arr_0"] 
	c = data["arr_1"][:, nset-1]
	b = data["arr_2"]-1
	labels = data["arr_3"]
	p_markov = data["arr_4"]
	bpred = data["arr_5"]

	print("p-value: " + str(p_markov[nset-1]))

	# filter cognitive states to remove jitters
	if Nsmooth != []:

		cf = np.zeros(c.shape)
		cf[0:Nsmooth] = c[0:Nsmooth]

		for n in range(Nsmooth, len(cf)):

			cf[n] = mode(c[n-Nsmooth:n])[0][0]

		c = cf

	# Compute cognitive-behavioral state-transitions

	if labels == []:

		labels = ['B'+str(int(z)+1) for z in np.unique(b)]

	states = [z1*10+z2 for z1 in np.unique(c) for z2 in np.unique(b)]

	if nset == 1:

		states_names = [str(labels[int(z2)]) for z2 in np.unique(b)]

	else:

		states_names = [str('C') + str(int(z1)) + ":"+ str(labels[int(z2)]) for z1 in np.unique(c) for z2 in np.unique(b)]
	

	Nstates = len(states)
	T = np.zeros((Nstates, Nstates))
	M = x.shape[0]

	for m in range(M-1):

		cur_sample = m
		next_sample = m+1

		cur_state = np.where(c[cur_sample]*10+b[cur_sample] == states)[0][0]
		next_state = np.where(c[next_sample]*10+b[next_sample] == states)[0][0]
		T[next_state, cur_state] += 1

	T = T / (M-1)
	T = T.transpose()

	# Draw graph

	if separate_plot == True:

		if save == False:

			plt.figure()
		
		else:

			fig = plt.figure(dpi = 400)


	G = nx.DiGraph()

	color_base = ['b','g','r','c','m','gray','y','peru']

	colors = nset * color_base[0:len(np.unique(b))]

	for n in range(len(states)):

		# if ('nostate' in states_names[n]) == False:

			G.add_node(str(states_names[n]))

	
	for n1 in range(Nstates):

		for n2 in range(Nstates):

			if (T[n1, n2] > T_thresh):# & (('nostate' in states_names[n1]) == False) & (('nostate' in states_names[n2]) == False):

				G.add_edge(states_names[n1], states_names[n2], weight = T[n1, n2] * max_width, color = colors[n1])


	# Positional layout design
	pos = nx.circular_layout(G)

	angles_for_cog_states = np.linspace(0, 2*np.pi, nset+1)[0:-1]

	# hack for better plotting
	
	if reorder != []:

		angles_for_cog_states = angles_for_cog_states[reorder]		


	amplitude_for_cog_states = np.ones(nset)
	cog_pos_cart = pol2cart([angles_for_cog_states, amplitude_for_cog_states])

	angles_for_behav_states = np.linspace(0, 2*np.pi, len(np.unique(b))+1)[0:-1]
	amplitude_for_behav_states = bspread*np.ones(len(np.unique(b)))
	behav_pos_cart = pol2cart([angles_for_behav_states, amplitude_for_behav_states])

	for n1 in range(nset):

		for n2 in range(len(np.unique(b))):

			poskey = states_names[(n1)*len(np.unique(b))+(n2+1)-1]
			pos[poskey] = [cog_pos_cart[0][n1] + behav_pos_cart[0][n2], cog_pos_cart[1][n1] + behav_pos_cart[1][n2]]

	# Plot graph
	edges = G.edges()
	weights = [G[u][v]['weight'] for u,v in edges]
	edge_colors = [G[u][v]['color'] for u,v in edges]
	node_sizes = np.diag(T) * max_size

	# fig.add_axes([0.25, 0.25, 0.5, 0.5])
	nx.draw(G, pos = pos, width = weights, edge_color = edge_colors, with_labels = True, arrowsize = size_arrows, connectionstyle="arc3,rad=-0.2", font_size = textsize, node_size = node_sizes, node_color = colors)

	if save == True:

		fig.savefig(file + "_graph_" + str(nset) + '_cog_states.png')  


	return str(p_markov[nset-1]), T, states_names


def plot_p_values_markov(save=False):

	#############
	#%% Imports #
	#############

	import matplotlib.pyplot as plt
	import matplotlib
	import numpy as np

	###############
	#%% Load data #
	###############

	p = np.zeros((20, 5))

	for n in range(5):

		file = "nc_mcm_model_worm_" + str(n+1) + ".npz"
		z = np.load(file)
		p_markov = z["arr_4"]
		p[:, n] = p_markov


	###########################
	#%% Plot individual worms #
	###########################

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(range(1,21), p, '.')
	ax.plot(range(1,21), np.mean(p, axis = 1), '-')
	ax.set_xticks(range(1,21))
	ax.set_yticks(np.arange(0, 1.1, 0.1))
	ax.set_xlim(2, 20.5)
	ax.set_ylim(0, 1.05)
	ax.set_xlabel('Number of cognitive states')
	ax.set_ylabel('p-value for rejecting Markovianity')
	ax.grid()
	ax.legend(['Worm 1', 'Worm 2', 'Worm 3', 'Worm 4', 'Worm 5', 'Average'])

	if save == True:

		fig.savefig('p_values_markov.png')


def plot_dec_process(filex, filey, nset, states, M = 20):

	#############
	#%% Imports #
	#############

	import matplotlib.pyplot as plt
	import numpy as np
	from matplotlib.lines import Line2D

	###############
	#%% Load data #
	###############

	data = np.load(filex)
	# x = data["arr_0"] 
	c = data["arr_1"][:, nset-1]
	b = data["arr_2"]-1
	labels = data["arr_3"]
	# p_markov = data["arr_4"]
	# bpred = data["arr_5"]

	data2 = np.load(filey)
	x = data2["arr_0"] 
	# b = data2["arr_2"]-1
	# labels = data["arr_3"]

	c = c[15:] # shorten data for bundle-net
	b = b[15:]

	########################
	# Extract trajectories #
	########################

	# get target states / behavs
	c1 = int(states[0][1])
	b1 = np.where(states[0][3::] == labels)[0][0]
	c2 = int(states[1][1])
	b2 = np.where(states[1][3::] == labels)[0][0]

	# find indices to target states
	i1 = np.where((b == b1) & (c == c1))[0] 
	i2 = np.where((b == b2) & (c == c2))[0]

	# extract first sample of each target state
	i1start = np.where(np.diff(i1) > 1)[0] + 1
	i2start = np.where(np.diff(i2) > 1)[0] + 1

	# extract M samples prior to each first sample
	K = x.shape[1]

	d1 = np.zeros((K, M+10, len(i1start)))
	dc1 = np.zeros((M+10, len(i1start)))

	for ii, im in enumerate(i1start):

		d1[:, :, ii] = x[ i1[im]-M+1:i1[im]+11, :].T
		dc1[:, ii] = c[ (i1[im]-M+1):i1[im]+11]


	d2 = np.zeros((K, M+10, len(i2start)))
	dc2 = np.zeros((M+10, len(i2start)))

	for ii, im in enumerate(i2start):

		d2[:, :, ii] = x[ i2[im]-M+1:i2[im]+11, :].T
		dc2[:, ii] = c[ (i2[im]-M+1):i2[im]+11]

	# plot
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	# colors = ['r','g','b','c','m','y']
	colors = [(n, 0, 0) for n in np.arange(0.5, 1, 0.5/len(i1start))]

	for i in range(len(i1start)):

		dplot = d1[:, :, i]

		# curax.arrow(0, 0, 0, -2.5, color = 'w', width=0, linewidth=0) # force offset to avoid overlap of behavioral legend with data
		for j in range(dplot.shape[1]-1):

			dx = 0.3*(dplot[0, j+1] - dplot[0, j])
			dy = 0.3*(dplot[1, j+1] - dplot[1, j])
			dz = 0.3*(dplot[2, j+1] - dplot[2, j])
			jcolor = colors[int(dc1[j,i]-1)]
			# ax.arrow(x[j, 0], x[j, 1], dx, dy, width=0.01, color = colors[int(i)], linewidth=0.01)
			ax.quiver(dplot[0, j], dplot[1, j], dplot[2, j], dx, dy, dz, linewidth = 1, length=2, normalize=False, color = jcolor, arrow_length_ratio=0.5)

	# colors = [(0, n, 0) for n in np.arange(0.5, 1, 0.5/len(i2start))]

	for i in range(len(i2start)):

		dplot = d2[:, :, i]

		# curax.arrow(0, 0, 0, -2.5, color = 'w', width=0, linewidth=0) # force offset to avoid overlap of behavioral legend with data
		for j in range(dplot.shape[1]-1):

			dx = 0.3*(dplot[0, j+1] - dplot[0, j])
			dy = 0.3*(dplot[1, j+1] - dplot[1, j])
			dz = 0.3*(dplot[2, j+1] - dplot[2, j])
			jcolor = colors[int(dc1[j,i]+2)]
			# ax.arrow(x[j, 0], x[j, 1], dx, dy, width=0.01, color = colors[int(i)], linewidth=0.01)
			ax.quiver(dplot[0, j], dplot[1, j], dplot[2, j], dx, dy, dz, linewidth = 1, length=2, normalize=False, color = jcolor, arrow_length_ratio=0.5)

	legend = [Line2D([0], [0], color=colors[n], lw=2) for n in range(len(colors))]	
	ax.legend(legend, ["C1 -> vt","C2 -> vt", "C3 -> vt", "C1 -> dt","C2 -> dt", "C3 -> dt"])


def plot_neuronal_state_differences(iworm, nset, states, save = True, tests = []):

	#############
	#%% Imports #
	#############

	import matplotlib.pyplot as plt
	import numpy as np
	from matplotlib.lines import Line2D

	###############
	#%% Load data #
	###############

	filex = "nc_mcm_model_worm_" + str(iworm) + ".npz"
	raw_file = "raw_full_data_worm_" + str(iworm) + ".npz"

	data = np.load(filex)
	# x = data["arr_0"] 
	c = data["arr_1"][:, nset-1]
	b = data["arr_2"]-1
	labels = data["arr_3"]
	# p_markov = data["arr_4"]
	# bpred = data["arr_5"]

	raw_data = np.load(raw_file)
	xraw = raw_data["arr_0"]

	##################
	# Prepare figure #
	##################

	if save == True:

		fig = plt.figure(dpi = 300, figsize = (16, 8))

	else:

		fig = plt.figure()
		# fig2 = plt.figure()


	ax = fig.add_subplot(1, 1, 1)

	########################
	# Extract trajectories #
	########################

	known_neurons = np.load("known_neurons.npz", allow_pickle=True)
	known_neuron_indices = known_neurons["arr_0"][iworm-1]
	known_neuron_names = known_neurons["arr_1"][iworm-1]

	K = len(states)
	xoffset = np.linspace(-0.225, 0.225, K)
	xwidth = 0.5/(K-1) - 0.05
	handles = []

	xpooled = np.empty((0, len(known_neuron_indices)))
	ipooled = np.empty((0))

	for k in range(0, K):

		# get target states / behavs
		c1 = int(states[k][1])
		b1 = np.where(states[k][3::] == labels)[0][0]
		# c2 = int(states[k][1])
		# b2 = np.where(states[k][3::] == labels)[0][0]

		# find indices to target states
		i1 = np.where((b == b1) & (c == c1))[0] 
		# i2 = np.where((b == b2) & (c == c2))[0]

		x1 = xraw[ np.ix_(i1, known_neuron_indices)]
		# x2 = xraw[ np.ix_(i2, known_neuron_indices)]

		x1m = np.mean( x1, axis = 0) # / np.std( x1, axis = 0)
		# x1m = np.std( x1, axis = 0)
		# x2m = np.mean( x2, axis = 0)

		# xstd = (np.std( xraw[ np.ix_(i1, known_neuron_indices)], axis = 0)*len(i1) + np.mean( xraw[ np.ix_(i2, known_neuron_indices)], axis = 0)*len(i2)) / (len(i1) + len(i2))
		# xstd = np.std(np.concatenate((x1, x2), axis = 0), axis = 0)

		# xd = (x1m - x2m) / xstd

		xpooled = np.concatenate((xpooled, x1), axis = 0)
		ipooled = np.concatenate((ipooled, k*np.ones(len(i1))), axis = 0)

		barhandle = ax.bar(range(len(known_neuron_indices)) + xoffset[k], x1m, width = xwidth)
		handles = handles + [barhandle]
		# ax2.hist(x1[:, 17])
		# ax.errorbar(range(len(known_neuron_indices)), x1m-x2m, yerr = xstd)

	fig.legend( handles, states, fontsize=20)
	ax.set_xlim([-1, len(known_neuron_indices)])
	ax.set_xticks(range(len(known_neuron_indices)))
	ax.set_xticklabels(known_neuron_names, rotation = 90, fontsize=18)
	# ax.grid()
	ax.set_ylabel('Mean neuronal activation in cognitive state', fontsize=20)
	fig.tight_layout()

	# Add stars for p-values
	if tests != []:

		iu = np.unique(ipooled)
		v = np.zeros((len(iu), xpooled.shape[1]))	

		for i in range(len(iu)):

			iv = np.where(ipooled == iu[i])[0]
			v[i, :] = np.mean(xpooled[iv, :], axis = 0)

		vm = np.max(v, axis = 0)

		for m in range(len(tests)):

			test = tests[m]

			test_indices = [n for n, nval in enumerate(ipooled) if nval in test]
			itest = ipooled[test_indices]
			xtest = xpooled[test_indices, :]

			T, p = permanova(xtest, itest)
			
			for k in range(len(p)):


				if p[k]*len(p) <= 0.01:

					ax.plot([k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9], [vm[k] + m*0.025 - 0.0075, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 - 0.0075], color = 'black', linewidth = 0.5)
					ax.text(k-0.15, vm[k] + m*0.025, "**")

				elif p[k]*len(p) <= 0.05:

					ax.plot([k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9], [vm[k] + m*0.025 - 0.0075, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 - 0.0075], color = 'black', linewidth = 0.5)
					ax.text(k-0.075, vm[k] + m*0.025, "*")

				else:

					pass


	if save == True:

		fig.savefig("diff_" + states[0] + states[1] + ".png")
		plt.close(fig)

	return xpooled, ipooled


def plot_cogstate_transition(iworm, nset, states, save = True):

	#############
	#%% Imports #
	#############

	import matplotlib.pyplot as plt
	import numpy as np
	from matplotlib.lines import Line2D

	###############
	#%% Load data #
	###############

	filex = "nc_mcm_model_worm_" + str(iworm) + ".npz"
	raw_file = "raw_full_data_worm_" + str(iworm) + ".npz"

	data = np.load(filex)
	# x = data["arr_0"] 
	c = data["arr_1"][:, nset-1]
	b = data["arr_2"]-1
	labels = data["arr_3"]
	# p_markov = data["arr_4"]
	# bpred = data["arr_5"]

	raw_data = np.load(raw_file)
	xraw = raw_data["arr_0"]

	##################
	# Prepare figure #
	##################

	if save == True:

		fig = plt.figure(dpi = 300, figsize = (8, 4))

	else:

		fig = plt.figure()


	ax = fig.add_subplot(1, 1, 1)

	########################
	# Extract trajectories #
	########################

	known_neurons = np.load("known_neurons.npz", allow_pickle=True)
	known_neuron_indices = known_neurons["arr_0"][iworm-1]
	known_neuron_names = known_neurons["arr_1"][iworm-1]

	c1 = int(states[0][1])
	b1 = np.where(states[0][3::] == labels)[0][0]
	c2 = int(states[1][1])
	b2 = np.where(states[1][3::] == labels)[0][0]

	i = [n for n in range(c.shape[0]-1) if (c[n] == c1) & (b[n] == b1) & (c[n+1] == c2) & (b[n+1] == b2)]
	j = [n for n in range(c.shape[0]-1) if (c[n] == c2) & (b[n] == b2) & (c[n+1] == c1) & (b[n+1] == b1)]

	print("#state transitions: " + str(len(i)) + " and " + str(len(j)))

	di = xraw[np.ix_(i, known_neuron_indices)]
	dj = xraw[np.ix_(j, known_neuron_indices)]

	dip = xraw[np.ix_([k+1 for k in i], known_neuron_indices)]
	djp = xraw[np.ix_([k+1 for k in j], known_neuron_indices)]

	h1 = ax.bar(np.arange(len(known_neuron_indices)) - 0.2, np.mean(di, axis = 0), width = 0.35, alpha = 0.5, color='blue')
	h2 = ax.bar(np.arange(len(known_neuron_indices)) - 0.2, np.mean(dip, axis = 0), width = 0.35,  alpha = 0.5, color = 'red')
	h3 = ax.bar(np.arange(len(known_neuron_indices)) + 0.2, np.mean(dj, axis = 0), width = 0.35, alpha = 0.5, color='blue')
	h4 = ax.bar(np.arange(len(known_neuron_indices)) + 0.2, np.mean(djp, axis = 0), width = 0.35, alpha = 0.5, color = 'green')

	ax.set_xlim([-1, len(known_neuron_indices)])
	ax.set_xticks(range(len(known_neuron_indices)))
	ax.set_xticklabels(known_neuron_names, rotation = 90)
	ax.set_ylabel('Neuronal activity at cognitive state transitions')
	ax.set_title(states[0] + "$\leftrightarrow$" + states[1])
	fig.tight_layout()

	# fig.legend( [h1, h3], states)
	if save == True:

		fig.savefig("dec_" + states[0] + states[1] + "_worm_" + str(iworm) + ".png")
		plt.close(fig)


def neuronal_perturbations(bfrom, bto, nset = 7, save = True, legend = []):

	import numpy as np
	import matplotlib.pyplot as plt

	# identify neurons common to all worms along with the indices for each worm
	known_neurons = np.load("known_neurons.npz", allow_pickle=True)
	known_neuron_indices = known_neurons["arr_0"]
	known_neuron_names = known_neurons["arr_1"]

	common_neurons = known_neuron_names[0]

	for n in range(4):

		common_neurons = np.intersect1d( common_neurons, known_neuron_names[n+1])

	neuron_indices = np.zeros((5, 22)).astype(int)

	for n1 in range(5):

		for n2 in range(22):

			ind = [m for m, mname in enumerate(known_neuron_names[n1]) if mname == common_neurons[n2]]
			neuron_indices[n1, n2] = known_neuron_indices[n1][ind]

	# get perturbations for each worm and cog state

	# plot
	if save == True:

		fig = plt.figure(dpi = 300, figsize = (8, 4))

	else:

		fig = plt.figure()

	ax = fig.add_subplot(1, 1, 1)
	xoffset = np.linspace(-0.25, 0.25, len(bto))

	perturb_all = np.empty((len(common_neurons), 0))
	cperturb_all = np.empty(0)
	vm = np.zeros((len(common_neurons), len(bto)))

	for icurbto, curbto in enumerate(bto):

		perturb_x = np.empty((len(common_neurons), 0))

		for iworm in range(5):

			# load data
			filex = "nc_mcm_model_worm_" + str(iworm+1) + ".npz"
			raw_file = "raw_full_data_worm_" + str(iworm+1) + ".npz"

			data = np.load(filex)
			x = data["arr_0"] 
			c = data["arr_1"][:, nset-1].astype(int)
			b = data["arr_2"]-1
			labels = data["arr_3"]
			# p_markov = data["arr_4"]
			# bpred = data["arr_5"]

			raw_data = np.load(raw_file)
			xraw = raw_data["arr_0"]

			# print(str(x.shape) + "_" + str(xraw.shape))

			# get all cognitive states with non-zero bfrom
			# for all above states, get their probability of transitioning into bto1 and bto2

			p, T, states = plot_ncmcmgraph("nc_mcm_model_worm_"+ str(iworm+1) + ".npz", nset, T_thresh=0, max_width=500, Nsmooth=[], separate_plot = True, reorder = [])		

			for istate, fromstate in enumerate(states):

				fromC, fromB = fromstate.split(":")

				if (fromB == bfrom) & (T[istate, istate] > 0): # if this is a revsus state with P > 0

					for jstate, tostate in enumerate(states): # iterate over all possible next states

						toC, toB = tostate.split(":") 

						if (toB == bfrom) & (T[istate, jstate] > 0): # if next state is revsus and can be reached

							prob_fromC_bto1 = np.sum([cond_transition_prob(T, states, [fromC + ":" + bfrom, "C" + str(n+1) + ":" + curbto]) for n in range(nset)])
							# prob_fromC_bto2 = np.sum([cond_transition_prob(T, states, [fromC + ":" + bfrom, "C" + str(n+1) + ":" + bto2]) for n in range(nset)])
							bratio_fromC = prob_fromC_bto1 #- prob_fromC_bto2

							prob_toC_bto1 = np.sum([cond_transition_prob(T, states, [toC + ":" + bfrom, "C" + str(n+1) + ":" + curbto]) for n in range(nset)])
							# prob_toC_bto2 = np.sum([cond_transition_prob(T, states, [toC + ":" + bfrom, "C" + str(n+1) + ":" + bto2]) for n in range(nset)])
							bratio_toC = prob_toC_bto1 # - prob_toC_bto2

							ib = np.where(labels == bfrom)[0][0]
							i1 = np.where((b == ib) & (c == int(fromC[1::])))[0]
							i2 = np.where((b == ib) & (c == int(toC[1::])))[0]

							dx = np.mean(xraw[ np.ix_(i1, neuron_indices[iworm, :])], axis = 0) - np.mean(xraw[ np.ix_(i2, neuron_indices[iworm, :])], axis = 0)
							dx_signed = dx * (bratio_toC - bratio_fromC)
							
							# from IPython import embed; embed()
							perturb_x = np.concatenate( (perturb_x, dx_signed[:, np.newaxis]), axis = 1)

							print("From " + str(fromC) + ":" + str(fromB) + " to " + str(toC) + ":" + str(toB) + ": " + str(bratio_fromC) + " vs " + str(bratio_toC))


		barhandle = ax.bar(range(len(common_neurons)) + xoffset[icurbto], np.mean(perturb_x, axis = 1), width = xoffset[-1])
		perturb_all = np.concatenate( (perturb_all, perturb_x), axis = 1)
		cperturb_all = np.concatenate( (cperturb_all, icurbto*np.ones(perturb_x.shape[1])), axis = 0)
		vm[:, icurbto] = np.mean(perturb_x, axis = 1)


	ax.set_xlim([-1, len(common_neurons)])
	ax.set_xticks(range(len(common_neurons)))
	ax.set_xticklabels(common_neurons, rotation = 90, fontsize=12)
	# ax.grid()
	ax.set_ylabel('Neuronal perturbations', fontsize=14)
	fig.legend( legend, fontsize=12)
	fig.tight_layout()

	# plot p-values
	T, p = permanova(perturb_all.T, cperturb_all)

	for k in range(len(p)):

		if p[k] <= 0.01 / len(p):

			# ax.plot([k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9], [vm[k] + m*0.025 - 0.0075, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 - 0.0075], color = 'black', linewidth = 0.5)
			ax.text(k -0.25, np.max(vm[k, :]), "++", fontsize=8)

		elif p[k] <= 0.05 / len(p):

			# ax.plot([k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9], [vm[k] + m*0.025 - 0.0075, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 - 0.0075], color = 'black', linewidth = 0.5)
			ax.text(k - 0.25, np.max(vm[k, :]), "+", fontsize=8)

		elif p[k] <= 0.01:

			# ax.plot([k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9], [vm[k] + m*0.025 - 0.0075, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 - 0.0075], color = 'black', linewidth = 0.5)
			ax.text(k - 0.25, np.max(vm[k, :]), "**", fontsize=8)

		elif p[k] <= 0.05:

			# ax.plot([k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[0]]-xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9, k+xoffset[test[-1]]+xwidth*0.9], [vm[k] + m*0.025 - 0.0075, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 + 0.005, vm[k] + m*0.025 - 0.0075], color = 'black', linewidth = 0.5)
			ax.text(k - 0.25, np.max(vm[k, :]), "*", fontsize=8)

		else:

			pass


	if save == True:

		filename = "perturbations_" + bfrom + ".png"
		fig.savefig(filename)
		plt.close(fig)


	return perturb_all, cperturb_all
 

######################
#%% helper functions #
######################


def cart2pol(cartcord):

	import numpy as np

	theta = np.arctan2(cartcord[1], cartcord[0])
	rho = np.hypot(cartcord[0], cartcord[1])

	return theta, rho


def pol2cart(polcoord):

	import numpy as np

	x = polcoord[1] * np.cos(polcoord[0])
	y = polcoord[1] * np.sin(polcoord[0])

	return x, y


def stdize(x):

	import numpy as np

	xmean = np.mean(x, axis = 0)
	xstd = np.std(x, axis = 0)
	xout = (x - xmean) / xstd

	return xout


def cond_transition_prob(T, states, query, cond = True):

	import numpy as np

	i1 = [n for n, state in enumerate(states) if state == query[0]][0]
	i2 = [n for n, state in enumerate(states) if state == query[1]][0]

	if cond == True:

		out = T[i1, i2] / np.sum(T[i1, :])

	elif cond == False:

		out = T[i1, i2]

	else:

		pass

	return out


def permanova(x, i, N = 10000):

	import numpy as np

	iu = np.unique(i)
	K = len(iu)

	V0 = np.var(x, axis = 0)

	Vk = np.zeros((K, x.shape[1]))
	lk = np.zeros(K)

	for k in range(K):

		j = np.where(i == iu[k])[0]
		Vk[k] = np.var(x[j, :], axis = 0) 
		lk[k] = len(j)

	T = np.dot(lk, Vk) / V0 / np.sum(lk)

	if N != []:

		TH0 = np.zeros((len(T), N))

		for n in range(N):

			xperm = np.zeros(x.shape)

			for m in range(x.shape[1]):

				xperm[:, m] = np.random.permutation(x[:, m])

			tH0, _ = permanova( xperm, i, N = [])
			TH0[:, n] = tH0

		p = np.mean(np.repeat(T[:, np.newaxis], N, axis = 1) >= TH0, axis = 1)

	else:

		p = []


	return T, p


def extract_all_known_neurons():

	from scipy.io import loadmat

	z = np.load("NoStim_Data.mat")


##############################################

