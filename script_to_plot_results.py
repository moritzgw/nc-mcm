#############################
# Import plotting functions #
#############################

run plotting_functions.py

#%% plot p-values
plot_p_values_markov(save=False)

#########################################################################
#%% plot behavioral motif comparison for worm 3 with 3 cognitive states #
#########################################################################

iworm = 3
nc = 3
N = 3
reorder = [2, 0, 1] # plotting order of cog states

fig = plt.figure(dpi = 300, figsize = (10, 5))
ax = fig.add_subplot(1,2,1)
plot_ncmcmgraph("nc_mcm_model_worm_" + str(iworm) + ".npz", 1, T_thresh=0.001, max_width=750, Nsmooth=N, separate_plot = False, textsize=12)
ax.set_title("(A) Behavioral transition diagram")

ax = fig.add_subplot(1,2,2)
plot_ncmcmgraph("nc_mcm_model_worm_" + str(iworm) + ".npz", nc, T_thresh=0.001, max_width=750, Nsmooth=N, separate_plot = False, reorder = reorder, bspread = 0.5, textsize=12)
ax.set_title("(B) Cognitive-behavioral transition diagram")

fig.tight_layout()
fig.savefig("bmotifs.png")
plt.close(fig)

###########################################################
#%% plot behavioral motifs of all worms for supp material #
###########################################################

N = 3 # smoothing parameter
nc_per_worm = [4, 4, 3, 3, 3]
reorder = 	[[2, 0, 1,3],
			[0, 3, 2, 1],
			[2, 0, 1],
			[2, 0, 1],
			[0, 1, 2]]
plot_width = 500
plot_size = 500
text_size = 6
thresh = 0.00075

fig = plt.figure(dpi = 300, figsize = (10, 5))

for n1 in range(5):

	ax = fig.add_subplot(2, 5, n1+1)
	plot_ncmcmgraph("nc_mcm_model_worm_" + str(n1+1) + ".npz", 1, T_thresh=thresh, max_width=plot_width, max_size=plot_size, Nsmooth=N, separate_plot = False, textsize = text_size)
	ax.set_title("Worm " + str(n1+1), fontsize = 8)
	ax = fig.add_subplot(2, 5, n1+6)
	plot_ncmcmgraph("nc_mcm_model_worm_" + str(n1+1) + ".npz", nc_per_worm[n1], T_thresh=thresh, max_width=plot_width, max_size=plot_size, Nsmooth=N, separate_plot = False, textsize = text_size, bspread=0.4, reorder = reorder[n1])
	# ax.set_title("p="p)

fig.tight_layout()
fig.savefig("bmotifs_all.png")
plt.close(fig)

#####################################################
#%% plot bundle-net manifolds with cognitive models #
#####################################################

N = 3 # smoothing parameter
nc_per_worm = [4, 4, 3, 3, 3]
reorder = 	[[2, 0, 1,3],
			[0, 3, 2, 1],
			[2, 0, 1],
			[2, 0, 1],
			[0, 1, 2]]
plot_width = 750
plot_size = 750
text_size = 10
thresh = 0.00075

for n in range(5):

	plot_trajectory("nc_mcm_model_worm_" + str(n+1) + ".npz", "bundle_net_results_consistent/bundlenet_consistent_embedding_worm_" + str(n) + ".npz", nc_per_worm[n], plottype='behavior', save = "bundlenet_w" + str(n) + ".png") 
	
	fig = plt.figure(dpi = 300, figsize = (5, 5))
	plot_ncmcmgraph("nc_mcm_model_worm_" + str(n+1) + ".npz", nc_per_worm[n], T_thresh=thresh, max_width=plot_width, max_size=plot_size, Nsmooth=N, separate_plot = False, textsize = text_size, bspread=0.4, reorder = reorder[n])
	# fig.tight_layout()
	fig.savefig("cognitive_w" + str(n) + ".png")
	plt.close(fig)	

###############################
#%% plot decision making info #
###############################

# worm 3 with 3 states
p, T, states = plot_ncmcmgraph("nc_mcm_model_worm_3.npz", 3, T_thresh=0, max_width=500, Nsmooth=N, separate_plot = True, reorder = [2, 0, 1])

np.sum([cond_transition_prob(T, states, ["C3:revsus","C" + str(n+1) + ":vt"]) for n in range(3)]) * 100
np.sum([cond_transition_prob(T, states, ["C1:revsus","C" + str(n+1) + ":vt"]) for n in range(3)]) * 100
np.sum([cond_transition_prob(T, states, ["C3:revsus","C" + str(n+1) + ":dt"]) for n in range(3)]) * 100
np.sum([cond_transition_prob(T, states, ["C1:revsus","C" + str(n+1) + ":dt"]) for n in range(3)]) * 100

# worm 3 with 7 states
p, T, states = plot_ncmcmgraph("nc_mcm_model_worm_3.npz", 7, T_thresh=0, max_width=500, Nsmooth=N, separate_plot = True, reorder = [2, 3, 6, 0, 1, 5, 4])

# perturbations
neuronal_perturbations("revsus", ["revsus","vt", "dt"], legend = ["$\Delta x^{revsus}_{revsus}$","$\Delta x^{revsus}_{vt}$","$\Delta x^{revsus}_{dt}$"])
neuronal_perturbations("slow", ["slow","fwd", "rev2"], legend = ["$\Delta x^{slow}_{slow}$","$\Delta x^{slow}_{fwd}$","$\Delta x^{slow}_{rev2}$"])

 