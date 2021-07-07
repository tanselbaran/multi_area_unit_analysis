# ETH - Institute of Neuroinformatics, Neurotechnology Group.
"""
Helper Functions for:
    Algorithm Designed for tracking Spike Clusters among sessions
    
Author: Samuel Ruipérez-Campillo
"""
import numpy                   as np
import scipy                   as sp
import h5py as h5
import matplotlib.pyplot       as plt
import seaborn                 as sns
import scipy.cluster.hierarchy as shc
from mpl_toolkits          import mplot3d
from matplotlib            import cm
from matplotlib.colors     import ListedColormap, LinearSegmentedColormap
from scipy import signal


##############################################################################
##############################################################################
## --------------------- ANALYTIC FUNCTIONS ------------------------------- ##
##############################################################################
##############################################################################  

def Extract_Param(files): 
    '''
    From the information extracted form the structured files (the lists of
    lists containing the Waveform, Spike Times and Labels), and from the 
    information about the sessions that want to be extracted from those files,
    Variables containing the Waveforms (and average waveforms for each cluster) 
    and Spike Times as well as cluster names are created. A reference time
    variable is also computed, using the input desired sampling. 

    Parameters
    ---------------------------------------------------------------------------
    sessions : array of int
        Indexes (in Data_3 and Labels_3) of the sessions from which the 
        waveforms want to be extracted.
        
    Data_3   : List of lists of lists ... of structures
        Contains all the data of waves and spike times, but needs to be 
        retrieved appropriately with the correct indexing and Labels.
        
    Labels_3 : Lists of lists of lists... of ints (or rarely strings)
        Contain in the last step the list with the number of the clusters
        that will be extracted from the appropriate session. It is used to 
        retrieve this data from Data_3.
        
    Sampling : int
        States one out of how many samples we want to use in the waveforms. It
        is important for the later dimensionality reduction, as each sample 
        is considered a 'parameter' or dimension. Thus, this Sampling variable
        determines the dimensions of our original space, where the observations
        (neuron spikes or waveforms) will be represented.

    Returns
    ---------------------------------------------------------------------------
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    Spike_Times : List of arrays of x floats 
        Contains the times (counted in samples, not in sec) when the spikes
        happened (x is the same as in Waveforms and av_Waveforms).
        
    clust_names : list of strings (that contain numbers)
        Strings with the name (identifier) of the clusters in the different 
        sessions
        
    Time        : Array of floats
        Uniformly distributed vector in a range of 2ms to serve as a reference
        for future plots. Its dimensions is the number of samples per spike
        divided by the Sampling variable. 

    '''

    waveforms = []
    spikeTimes = []
    clust_names = []
    
    for file in files:
        print(file)        
        if file[-4:] == '.mat':
            data = h5.File(file,'r')
            sortedSpikeClusters = np.asarray(data['sortedSpikeClusters'],'int16')
            clusterNotes = np.asarray(data['clusterNotes'])
            spikesFiltFull = np.asarray(data['spikesFiltFull'],'int16')
            spikesFiltFull = spikesFiltFull[:,:,::3]
            spikeTimesSorted = np.asarray(data['spikeTimesSorted'])
            
            for j, clus in enumerate(clusterNotes[0]):
                obj = data[clus]
                clusterNote = ''.join(chr(k) for k in obj[:])
                if clusterNote == 'single':
                    clusterId = j+1
                    clusterSpikes = np.where(sortedSpikeClusters == clusterId)[1]
                    clusterWaveforms = spikesFiltFull[clusterSpikes]
                    clusterSpikeTimes = spikeTimesSorted[0][clusterSpikes]
                    clust_names.append(file[:-4]+'_'+str(clusterId))
                    waveforms.append(clusterWaveforms)
                    spikeTimes.append(clusterSpikeTimes)
    
    Time = np.linspace(0,2,np.int(60)) # Time Reference for all the spikes (samples to time conversion)             
    return waveforms, spikeTimes, clust_names, Time


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def extract_HAwav_t(Option, n_chan, Waveforms, av_Waveforms, Clust_ref):
    '''
    From the average waveforms of all the channels of all the clusters, this 
    function extracts the highest amplitude n channels (spikes) of a reference 
    cluster and the spike waves in these channels for the other clusters (i.e.
    spatial information is maintained as same channels are selected for all 
    clusters). It also returns the ordered indices of such channels.

    Parameters
    ---------------------------------------------------------------------------
    n_chan       : int
        Number of channels that are going to be concatenated.
        
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    Clust_ref    : int
        Cluster taken as a reference to decide the number of channels with
        highest Amplitude.
    
    Returns
    ---------------------------------------------------------------------------
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in the case of the reference cluster, the channels where
        the highest amplitude spikes are found).
    
    indices_maxn   : List of 1 element with an array of n elements
        Contains the indices with the maximum amplitude of the reference. n is
        the number of indices (i.e. n_chan).
    '''
    
    n = n_chan         # Number of indexes that want to be saved
    Waveforms_sum = [] # List with the components of the absolute sum of each av. spike channel for reference cluster.
    indices_maxn  = [] # List of channels ordered by their highest average spike amplitude (dicrete integral).
    Clust_ref     = 0  # Select the reference cluster (with respect to the temporal aligning)
    Channels_sum  = []
    for j in range(av_Waveforms[Clust_ref].shape[0]): # For each Channel
        Channels_sum.append(np.sum(np.abs(av_Waveforms[Clust_ref][j]))) # Append the abs sum of each channel.  
    
    Channels_sum = np.array(Channels_sum)
    idx_maxn = [] 
    idx_maxn = np.argpartition(Channels_sum, -n)[-n:] # indexes of max amplitude - ordered (smallest A. - biggest A.)
    
    indices_maxn.append(idx_maxn[np.argsort((-Channels_sum)[idx_maxn])])  # Append channel index w\ highest spike amplitude.
    Waveforms_sum.append(Channels_sum) # Append the abs sum of all the channels for each cluster.
    
        # Concatenate spikes (samples) of channels with highest amplitude.
    Waveforms_HA   = [] # List of lists with samples from n spikes (temorally aligned to HA of Clust_ref)
    Waveforms_HA_c = [] # Concatenated n channels of waveforms
    for j in range(len(Waveforms)): # For every Cluster
        Waveforms_HA_channel = []   # List of n temporally aligned spikes for channel j
        for i in range(n):          # For every Channel n
            Waveforms_HA_channel.append(Waveforms[j][:,indices_maxn[0][i],:])
        Waveforms_HA.append(Waveforms_HA_channel)
        Waveforms_HA_c.append(np.concatenate(Waveforms_HA[j], axis = 1)) # Concatenated n channels of waveforms
    
    return Waveforms_HA_c, indices_maxn


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################

def extract_HAwav(n_chan, Waveforms, av_Waveforms):
    '''
    From the average waveforms of all the channels of all the clusters, this 
    function extracts the highest amplitude n channels (spikes) of all the 
    clusters (i.e. the spatial information is NOT maintained as NOT same 
    channels are selected for all clusters). It also returns the ordered 
    indices of such channels.

    Parameters
    ---------------------------------------------------------------------------
    n_chan       : int
        Number of channels that are going to be concatenated.
        
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    
    Returns
    ---------------------------------------------------------------------------
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in all cases, the channels where the highest amplitude 
        spikes are found).
        
    indices_maxn   : List of 1 element with an array of n elements
        Contains the indices with the maximum amplitude of the reference. n is
        the number of indices (i.e. n_chan)
    '''  
    
    n = n_chan          # Number of indexes that want to be saved
    Waveforms_sum = [] # List with the components of the absolute sum of each av. spike channel for each cluster.
    indices_maxn  = [] # List of channels ordered by their highest average spike amplitude (dicrete integral).
    for i in range(len(av_Waveforms)): # For each Cluster
        Channels_sum = []
        for j in range(av_Waveforms[i].shape[0]): # For each Channel
            Channels_sum.append(np.sum(np.abs(av_Waveforms[i][j]))) # Append the abs sum of each channel.  
        
        Channels_sum = np.array(Channels_sum)
        idx_maxn = [] 
        idx_maxn = np.argpartition(Channels_sum, -n)[-n:]
        
        indices_maxn.append(idx_maxn[np.argsort((-Channels_sum)[idx_maxn])])  # Append channel index w\ highest spike amplitude.
        Waveforms_sum.append(Channels_sum) # Append the abs sum of all the channels for each cluster.
    
    # Concatenate spikes (samples) of channels with highest amplitude.
    Waveforms_HA   = [] # List of lists with samples from n highest amplitude spikes
    Waveforms_HA_c = [] # Concatenated n channels of waveforms
    for j in range(len(Waveforms)): # For every Cluster
        Waveforms_HA_channel = [] # List of n highest amplitude spikes for channel j
        for i in range(n): # For every Channel n
            Waveforms_HA_channel.append(Waveforms[j][:,indices_maxn[j][i],:])
        Waveforms_HA.append(Waveforms_HA_channel)
        Waveforms_HA_c.append(np.concatenate(Waveforms_HA[j], axis = 1)) # Concatenated n channels of waveforms
    
    return Waveforms_HA_c, indices_maxn


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def extract_HAwav_av(Option, n_chan, Waveforms_HA_c):
    '''
    In case of working on the modes Option 2 or 3, it returns the average 
    waveform of the concatenated channels of higher amplitude.

    Parameters
    ---------------------------------------------------------------------------
    Option         : int (1, 2 or 3)
        Mode of operation
        
    n_chan         : int
        Number of channels of the recording system (i.e. electrodes)
        
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in all cases, the channels where the highest amplitude 
        spikes are found).
    
    Returns
    ---------------------------------------------------------------------------
    Waveforms_HA_c_av : list of x elements, of dimensions [q,]
        Contains the average value of the concatenation of all the channels 
        selected for each of the clusters (the average is taken from all the
        observations of spike waveforms)
        q : number of dimensions: samples of concatenated spikes (60 x n, with 
        n as number of waves/channels)
    '''  
    
    if Option == 2 or Option == 3:
        Waveforms_HA_c_av = []
        # Time_2 = Time = np.linspace(0,2*n_chan,Waveforms_HA_c[0].shape[1])
        for i in range(len(Waveforms_HA_c)):
            Waveforms_HA_c_av.append(np.mean(Waveforms_HA_c[i], axis = 0)) # Av Waveform for each cluster.
    
    return Waveforms_HA_c_av


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################

def extract_Allwav(Waveforms):
    '''
    From the average waveforms of all the channels of all the clusters, this 
    function extracts all the channels from all the clusters (i.e. the spatial 
    information is maintained as all clusters as selected). 

    Parameters
    ---------------------------------------------------------------------------
        
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    
    Returns
    ---------------------------------------------------------------------------
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in all cases, the channels where the highest amplitude 
        spikes are found).
        
    '''  
    
    Waveforms_HA_c = []
    for i in range(len(Waveforms)): # For each cluster of spikes
        Waveforms_HA_c.append(np.reshape(Waveforms[i], [Waveforms[i].shape[0],Waveforms[i].shape[1]*Waveforms[i].shape[2]],'C'))

    return Waveforms_HA_c


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def conc_Allsamp(Waveforms_HA_c):
    '''
    From the concatenated waveforms for each spike for every cluster, all
    spikes are joint in a variable in a n-Dimensional space (with n number of
    sample points in each of the concatenated waveform)

    Parameters
    ---------------------------------------------------------------------------
        
    Waveforms_HA_c : list of x elements, of dimensions [p,q]
        Contains the concatenation of the all channels, for each of the
        clusters.
        p : number of spikes for each of the clusters (different in each case).
        q : number of dimensions: samples of concatenated spikes (60 x n, with 
        n as number of waves/channels)
        
    Returns
    ---------------------------------------------------------------------------
    Waveforms_Allsamp : list of x elements, of dimensions [p,q]
        Contains the concatenation of the 5 channels selected, for each of the
        clusters (e.g. in all cases, the channels where the highest amplitude 
        spikes are found).
        
    '''  
    
    Waveforms_Allsamp = Waveforms_HA_c[0]
    for i in range(1,len(Waveforms_HA_c)):
        Waveforms_Allsamp = np.concatenate((Waveforms_Allsamp, Waveforms_HA_c[i]), axis = 0)    
    return Waveforms_Allsamp

##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def compute_distmat(Proj, metric):
    '''
    From the Projection matrices, that is, the matrices containing the points
    projected into the Principal-Components Subspace previously designed, the 
    distance in such subspace is measured between clusters belonging to 
    different spike clusters.

    Parameters
    ---------------------------------------------------------------------------
    Proj    : List of x elements with dimensions [m,n]
        List of the projections matrices. Each element x_i of the list contains
        one matrix representing the clusters of all samples (spikes) 
        projected (reduced) to the Principal-Components subspace. The dimensions
        of each of these matrices is [m,n]:
        m : number of dimensions for each sample (spike), i.e. number of PC axes
        n : number of samples (spikes) per cluster
        
    metric  : String
        String containing the metric. We assume so far that it is mahalanobis.
    
    Returns
    ---------------------------------------------------------------------------
    Av_dist_mat : np.array matrix of dimensions [q,q]
        Contains the average (mahalanobis) distance between clusters. The 
        diagonal represents the distance of one cluster with itself, and the
        rest of columns, the distance of such cluster to all the other clusters.
        q : number of clusters (of spikes).
    
    Side Notes
    ---------------------------------------------------------------------------
    Important Note I: Note that the distance of each cluster with itself is not
        zero. This is because the distance is computed as the average distance 
        of each point to the rest of points. Therefore, one point in its cluster
        is nearer to the rest of points of the cluster, so the distance is 
        shorter but not zero.
        
    Important Note II: In order to compute the distance of a cluster to itself, 
        LOO (Leave-one-out) algorithm is utilised, meaning that the point being
        evaluated is taken out of the distribution of its cdist vs pdistown cluster, when
        measuring the mahalanobis distance of such point to the rest of the 
        points of its cluster. Thence, biases are avoided.
    '''  
    
    Av_dist = np.zeros((len(Proj),len(Proj))) # List of average distance from every point in cluster j to every cluster i. Diagonal!!!       
    
    cluster_means = np.zeros((len(Proj),len(Proj[0])))
    var = np.zeros((len(Proj), len(Proj[0])))
    
    for i, cluster in enumerate(Proj):
        cluster_means[i] = np.mean(cluster,1)
        var[i] = np.var(cluster,1)
    
    for i, cluster1 in enumerate(Proj): # For every cluster 
        print(i)
        cluster_dists = np.zeros((len(cluster1[0]),len(Proj)))
        
        for j,point in enumerate(cluster1.T):
            for k,cluster2 in enumerate(Proj):
                cluster_dists[j,k] = sp.spatial.distance.seuclidean(point,cluster_means[k],var[k])
        Av_dist[i] = np.mean(cluster_dists,0)

    return Av_dist

##############################################################################
##############################################################################
## ------------------------ DISPLAY FUNCTIONS ----------------------------- ##
##############################################################################
##############################################################################
    
def PlotSingleWaveforms(Waveforms, av_Waveforms, Time, cl):
    '''
    Plot a set of Waveform spikes in different figures

    Parameters
    ---------------------------------------------------------------------------
    Waveforms    : List with x elements, of dimensions [n,m,s]
        List with all waveforms for each of the clusters:
        x : number of clusters
        n : number of spikes recorded for each of the clusters (samples)
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    av_Waveforms : List with x elements, of dimensions [m,s]
        Average waveforms for each of the clusters in all the clusters:
        x : number of clusters
        m : number of channels
        s : number of samples per spike (i.e. number of samples in each channel 
        of av. Waveform)
        
    Time        : Array of floats
        Uniformly distributed vector in a range of 2ms to serve as a reference
        for future plots. Its dimensions is the number of samples per spike
        divided by the Sampling variable.
    cl           : int
        Cluster that is meant to be used for the plots
        
    Returns
    ---------------------------------------------------------------------------
    No variable returned: A plot of the Waveforms is the output.

    '''
    font = {'family' : 'Times New Roman',
                'weight' : 'bold',
                'size'   : 15}
    
    for i in range(Waveforms[cl].shape[1]):
        fig, ax = plt.subplots()
        namefig = 'Cluster ' + str(cl) + ' channel ' + str(i) +'.pdf'
        namefig2 = 'Cluster ' + str(cl) + ' channel ' + str(i) +'.svg'
        plt.xlabel('Time (sec)', fontfamily = 'Times New Roman', fontsize = 18)
        plt.ylabel('Amplitude (mV)', fontfamily = 'Times New Roman', fontsize = 18)
        plt.title(namefig[0:-4], fontfamily = 'Times New Roman', fontsize = 18)
        for j in range(Waveforms[cl].shape[0]):
            plt.plot(Time,Waveforms[cl][j,i,:],c ='darkgrey', linewidth = 0.1)
        plt.plot(Time,av_Waveforms[cl][i,:],   c = 'black',   linewidth = 2)
        ax.set(xlim=(Time[0], Time [-1]))
        plt.rc('font', **font)
        plt.grid(b=bool, which='major', axis='both', color='gray', linestyle='--', linewidth=0.6)
        plt.minorticks_on()
        plt.grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
        
        plt.savefig(namefig)
        plt.savefig(namefig2)
        
##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
        
def PlotWaveforms(Print_S_Opt, Option, Waveforms_HA_c, Waveforms_HA_c_av, Time_2, n_chan, indices_maxn):
    '''
    This function concatenates the spike waves from all the channels taken into
    account from each of the clusters (i.e. for each of the neuronal activities),
    and plot different figures with that information.

    Parameters
    ---------------------------------------------------------------------------
    Print_S_Opt       : int (2 or 3)
        Printing mode - All clusters separately and together (2) or only one 
        figure with all clusters together (3).
    Option            : int (1, 2 or 3)
        Mode of operation
        
    Waveforms_HA_c    : list of x elements, of dimensions [p,q]
        Contains the concatenation of the all channels, for each of the
        clusters.
        p : number of spikes for each of the clusters (different in each case).
        q : number of dimensions: samples of concatenated spikes (60 x n, with 
        n as number of waves/channels)
        
    Waveforms_HA_c_av : list of x elements, of dimensions [q,]
        Contains the average value of the concatenation of all the channels 
        selected for each of the clusters (the average is taken from all the
        observations of spike waveforms)
        q : number of dimensions: samples of concatenated spikes (60 x n, with 
        n as number of waves/channels)
        
    Time              : Array of floats
        Uniformly distributed vector in a range of 2ms to serve as a reference
        for future plots. Its dimensions is the number of samples per spike
        divided by the Sampling variable. 
        
    n_chan            : int
        Number of channels that are going to be concatenated.
        
    indices_maxn      : List of 1 element with an array of n elements
        Contains the indices with the maximum amplitude of the reference. n is
        the number of indices (i.e. n_chan)

    Returns
    ---------------------------------------------------------------------------
    No variable returned: A plot of the concatenated waveforms for all clusters
    is the output.

    ''' 
           
    font = {'family' : 'Times New Roman',
                'weight' : 'bold',
                'size'   : 15}
    
    if Print_S_Opt == 2: # Plot every single concatenation of spikes
        for i in range(len(Waveforms_HA_c)):
            f, ax = plt.subplots()
            plt.suptitle('Temporary aligned HA Spikes', fontfamily = 'Times New Roman', fontsize = 22)
            plt.xlabel('Concatenated Channels (Time in sec - reference)', fontfamily = 'Times New Roman', fontsize = 18)
            plt.ylabel('Amplitude (mV)', fontfamily = 'Times New Roman', fontsize = 18)
            
            samp = Waveforms_HA_c[i].shape[0]//200 + 1
            rep  = np.linspace(0,Waveforms_HA_c[i].shape[0],Waveforms_HA_c[i].shape[0]//samp)
            rep_int = [np.int(x) for x in rep]
            
            for j in rep_int[:-1]:
                plt.plot(Time_2,Waveforms_HA_c[i][j,:], c ='darkgrey', linewidth = 0.1)
            plt.plot(Time_2,Waveforms_HA_c_av[i],       c = 'black',   linewidth = 2)
            
            for xc in range(n_chan):
                if Option == 2:
                    plt.axvline(x = xc*2, c = 'darkgrey' , ls = ':', lw = 2, linewidth = 1.5, label = 'Channel number: ' + str(indices_maxn[0][xc]))
                elif Option == 3: 
                    plt.axvline(x = xc*2, c = 'darkgrey' , ls = ':', lw = 2, linewidth = 1.5)
            plt.axvline(x = xc*2+2, c = 'darkgrey' , ls = ':', lw = 2, linewidth = 1.5)
            if Option == 2:
                plt.legend(loc = 'lower right')
            namefig = 'Cluster_' + str(i) + '.pdf'
            namefig2 = 'Cluster_' + str(i) + '.svg'
            
            # ax.set(xlim=(Time[0], Time [-1]))
            plt.rc('font', **font)
            plt.grid(b=bool, which='major', axis='both', color='gray', linestyle='--', linewidth=0.6)
            plt.minorticks_on()
            plt.grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
            
            plt.savefig(namefig)
            plt.savefig(namefig2)
        
        # Plot all clusters in the same figure
    fig, axs = plt.subplots(len(Waveforms_HA_c), sharex = True, sharey = False)
    plt.suptitle( 'Temporary aligned HA Spikes', fontweight = 'heavy', fontfamily = 'Times New Roman', fontsize = 18)
    plt.xlabel('Time (sec)', fontfamily = 'Times New Roman')
    fig.text(0.04, 0.5, 'Amplitude (mV)', va='center', ha='center', rotation='vertical', fontfamily = 'Times New Roman')
    for i in range(len(Waveforms_HA_c)):
        
        samp = Waveforms_HA_c[i].shape[0]//150 + 1
        rep  = np.linspace(0,Waveforms_HA_c[i].shape[0],Waveforms_HA_c[i].shape[0]//samp)
        rep_int = [np.int(x) for x in rep]
            
        for j in rep_int[:-1]:
            axs[i].plot(Time_2,Waveforms_HA_c[i][j,:], c ='darkgrey', linewidth = 0.08)
        axs[i].plot(Time_2,Waveforms_HA_c_av[i],       c = 'black',   linewidth = 1)
        for xc in range(n_chan):
            axs[i].axvline(x = xc*2, c = 'dimgray' , ls = ':', lw = 2, linewidth = 1)
        axs[i].axvline(x = xc*2+2, c = 'dimgray' , ls = ':', lw = 2, linewidth = 1)
        
        axs[i].grid(b=bool, which='major', axis='both', color='gray', linestyle='--', linewidth=0.6)
        axs[i].minorticks_on()
        axs[i].grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
        print('Plotting Waveforms: Cluster ' + str(i))
    
    f = plt.gcf()  # f = figure(n) if you know the figure number
    f.set_size_inches(8.27,11.69)
        
    if len(indices_maxn) == 0:
        t = np.linspace(0,42,43)
        t_Names = [
                    '', 'Channel 0',  '', 'Channel 1',  '', 'Channel 2', 
                    '', 'Channel 3',  '', 'Channel 4',  '', 'Channel 5',
                    '', 'Channel 6',  '', 'Channel 7',  '', 'Channel 8',
                    '', 'Channel 9',  '', 'Channel 10', '', 'Channel 11',
                    '', 'Channel 12', '', 'Channel 13', '', 'Channel 14',
                    '', 'Channel 15', '', 'Channel 16', '', 'Channel 17',
                    '', 'Channel 18', '', 'Channel 19', '', 'Channel 20', ''
                    ]

        plt.xticks(t, t_Names, rotation = 70, fontsize = 5)
    
    namefig = 'Alltogether' + '.pdf'
    namefig2 = 'Alltogether' + '.svg'
    fig.savefig(namefig)
    fig.savefig(namefig2)

##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
            
def ClustPlot(n_pca, Proj):
    '''
    The current function takes the number of PCA components and
    
    Parameters
    --------------------------------------------------------------------------
    n_pca : int
        Number of Principal-Components (that is, axis in our projection sub-
        space).
    Proj  : list n elements pxq
        Represents the clusters of spikes in the projection subspace.
        n: Number of clusters that have been projected in the p-dimensional space.
        p: dimensions of the subspace.
        q: number of samples or observations (spikes) for each neuron cluster.

    Returns
    --------------------------------------------------------------------------
    ax1 : fig object
        Figure File of the 3D projection with initial view 1.
    ax2 : fig object
        Figure File of the 3D projection with initial view 2.
    ax2 : fig object
        Figure File of the 3D projection with initial view 3.

    '''
    
    cmaps = [
        'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
        'hot', 'afmhot', 'gist_heat', 'copper', 'spring', 'summer', 
        'autumn', 'winter', 'cool', 'Wistia', 'binary', 'gist_yarg', 
        'gist_gray', 'gray', 'bone', 'pink'
    ]   # List of Colour maps.
    
    font = {'family' : 'Times New Roman',
                'weight' : 'bold',
                'size'   : 15}
    
    if n_pca == 3:
            # Plot Figures in different views of the 3D projection
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.view_init(60, 125)    
        for i in range(len(Proj)):
            samp = Proj[i].shape[1]//300 + 1
            print(samp)
            ax.scatter3D(Proj[i][0,0::samp], Proj[i][1,0::samp], Proj[i][2,0::samp], cmap = cmaps[i], alpha = 0.3,linewidths = 0.7) 
        
    if n_pca == 2:
        fig = plt.figure()
        ax = plt.axes
        for i in range(len(Proj)):
            samp = Proj[i].shape[1]//300 + 1
            fig = plt.scatter(Proj[i][0,0::samp], Proj[i][1,0::samp], cmap = cmaps[i], alpha = 0.3,linewidths = 0.4)
        plt.grid(b=bool, which='major', axis='both', color='silver', linestyle='--', linewidth=0.6)
        # plt.grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
        
    plt.xlabel('X', fontfamily = 'Times New Roman', size = 20)
    plt.ylabel('Y', fontfamily = 'Times New Roman', size = 20)
    Tit = 'PC-Subspace with ' + str(n_pca) + ' Components'
    plt.title(Tit, fontfamily = 'Times New Roman')
    plt.rc('font', **font)
    
    name_fig = Tit + '.pdf'
    name_fig2 = Tit + '.png'
    name_fig3 = Tit + '.svg'
    
    plt.savefig(name_fig)
    plt.savefig(name_fig2)
    plt.savefig(name_fig3)
    
    return ax


##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
    
def plotMahala_mat(Av_dist_mat, Option, Suboption, n_chan, Tot_Var):
    '''
    

    Parameters
    ---------------------------------------------------------------------------
    Av_dist_mat : np.array matrix of dimensions [q,q]
        Contains the average (mahalanobis) distance between clusters. The 
        diagonal represents the distance of one cluster with itself, and the
        rest of columns, the distance of such cluster to all the other clusters.
        q : number of clusters (of spikes).
        
    Option      : int (1, 2 or 3)
        Mode of operation
        
    Suboption   : int
        0 or 1 depending on if the n_pca is taken into account (0) or perc_pca(1)
        
    n_chan      : int
        Number of channels that are going to be concatenated.
        
    Tot_Var     : float
        Percentage of information or variance maintained in the PC-subspace.

    Returns
    ---------------------------------------------------------------------------
    None.

    '''
    font = {'family' : 'Times New Roman',
                'weight' : 'bold',
                'size'   : 7}
    
    Tit = 'Mah_Dist_w_' + 'Opt_' + str(Option) + '_Subopt_' + str(Suboption) + '_NChan_' + str(n_chan) + '_VarPCA_' + str("{:.2f}".format(Tot_Var))
    f = plt.figure()
    plt.title(Tit, fontfamily = 'Times New Roman')
    sns.heatmap(Av_dist_mat, annot = True, cmap = 'gray', linewidths=0.5, annot_kws={"size": 6})
    plt.xlabel('Cluster Number', fontfamily = 'Times New Roman', size = 20)
    plt.ylabel('Cluster Number', fontfamily = 'Times New Roman', size = 20)
    name_fig = Tit + '.pdf'
    name_fig2 = Tit + '.svg'
    plt.rc('font', **font)
    f.savefig(name_fig)
    f.savefig(name_fig2)
    

##############################################################################
# -------------------------------------------------------------------------- #
##############################################################################
        
def plot_hierarchical_dend(Red_mat, hier_method):
          
    Link_mat = shc.linkage(Red_mat, method = hier_method, metric = 'euclidean', optimal_ordering = False)
    
    fig = plt.figure()
    dend = shc.dendrogram(Link_mat)
    plt.xlabel('Cluster Number', fontfamily = 'Times New Roman')
    plt.ylabel('Cluster Distance Metric', fontfamily = 'Times New Roman')
    Tit = 'Hierarchical Linkage Clustering with ' + str(hier_method) + ' Method'
    plt.title(Tit, fontfamily = 'Times New Roman')
    Link = shc.linkage(Red_mat, hier_method)
    plt.grid(b=bool, which='major', axis='both', color='gray', linestyle='--', linewidth=0.6)
    plt.minorticks_on()
    plt.grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
    
    # fig.grid(b=bool, which='minor', axis='both', color='silver', linestyle='--', linewidth=0.3)
    # minorticks_on()
    name_fig = Tit + '.pdf'
    name_fig2 = Tit + '.svg'
    fig.savefig(name_fig)
    fig.savefig(name_fig2)
    print('Plotting Hierarchical Dendrogram: Type ' + hier_method)
    return Link
    
