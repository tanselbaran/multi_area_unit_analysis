# ETH - Institute of Neuroinformatics, Neurotechnology Group.
"""
Algorithm Designed for tracking Spike Clusters among sessions
Author: Samuel Ruipérez-Campillo

Version 3 - Cool Results with sample clusters from first sessions. Mahalanobis
            has been implemented and also has the PCA Analysis been.
"""

import numpy             as np
import matplotlib.pyplot as plt
import h5py              as h5
from joblib import parallel_backend
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

import os
from scipy.spatial.distance import pdist
import scipy.io
from scipy import signal

folder = os.getcwd()
import Track_Functions   as TF
os.chdir('/home/baran/Desktop/mouseLL2_spike_sorting_results/_sorted_spike_data')
# %% Extract Spikes from Structured Data file

files = os.listdir()
files.sort()
ds_factor = 2

Waveforms, Spike_Times, clust_names, Time = TF.Extract_Param(files)
Suboption   = 1    # Choose: 0 - number of components of the PCA Analysis is constant (e.g. 3)
                   #         1 - number of components of PCA depends on the amount of variance desired (e.g. 60%)

Print_S_Opt = 0    # Choose: 0 - No print
                   # Choose: 1 - plot a set of single spikes in different figures
                   # Choose: 2 - Plot concatenated waveforms: Each Single Cluster and All Together
                   # Choose: 3 - Plot concatenated waveforms: Only all Clusters Together

n_pca    = 3     # Number of Principal components if Suboption = 0
perc_pca = 0.9  # Percentage of variance in pca  if Subotpion = 1

# Extract n = all channels with highest amplitude from the averages.
n_chan            = len(Waveforms[0][0][0]) # Number of Channels
Waveforms_HA_c    = TF.extract_Allwav(Waveforms) # Concatenated Waves
Waveforms_Allsamp = TF.conc_Allsamp(Waveforms_HA_c) # All Spike-waves

# Create PC-Subspace and project clusters onto it.

pca = PCA(n_components=perc_pca)
scaler = MaxAbsScaler()

print('Fitting PCA')
with parallel_backend('threading', n_jobs=6):
    Waveforms_Allsamp_standardized = scaler.fit_transform(Waveforms_Allsamp)
    Proj = pca.fit(Waveforms_Allsamp_standardized)


Proj = []
for i in range(len(Waveforms_HA_c)):
    print("Transforming cluster:"+str(i))
    with parallel_backend('threading', n_jobs=6):
        Waveforms_HA_c_standardized = scaler.fit_transform(Waveforms_HA_c[i])
        Proj.append(np.transpose(pca.transform(Waveforms_HA_c_standardized)))

# Plot Clusters if we have 2- or 3-D space
#if Suboption == 0:
#    ax = TF.ClustPlot(n_pca, Proj)

# %% Data from PCA and PC-Subspace

Variance_Ratio = pca.explained_variance_ratio_
S_Values       = pca.singular_values_
Tot_Var        = np.sum(Variance_Ratio)

# %% SPIKES PLOTS

if Print_S_Opt == 0:   # No Plot
    print('No spike waveforms plotted.')

elif Print_S_Opt == 1: # Plot a set of single spikes in different figures
    TF.PlotSingleWaveforms(Waveforms, av_Waveforms, Time, cl = 0)

elif Print_S_Opt == 2 or Print_S_Opt == 3: # Plot concatenated waveforms
    Time = np.linspace(0,2*n_chan,Waveforms_HA_c[0].shape[1])
    Waveforms_HA_c_av = TF.extract_HAwav_av(Option, n_chan, Waveforms_HA_c) # Av. Waveforms
    if Option == 3: # No ref cluster, no indexes
        indices_maxn = []
    TF.PlotWaveforms(Print_S_Opt, Option, Waveforms_HA_c, Waveforms_HA_c_av, Time, n_chan, indices_maxn)

else: # Invalid option
    print('Please select a valid option for plotting spike waveforms.')

# Compute the Mahalanobis distance of every point to every distribution,
# excluding the point from its own distribution to avoid biases (LOO Algorithm).

# Compute Metric
Av_dist_mat = TF.compute_distmat(Proj, metric = 'mahalanobis')
Av_dist_mat_sim = (Av_dist_mat + Av_dist_mat.transpose())/2
Red_mat = pdist(Av_dist_mat_sim)

# Plot Mahalanobis Distance Matrices
#TF.plotMahala_mat(Av_dist_mat_sim, Option, Suboption, n_chan, Tot_Var)

Link_ward   = TF.plot_hierarchical_dend(Red_mat, hier_method = 'ward')
np.save('Link_ward.npy',Link_ward)
