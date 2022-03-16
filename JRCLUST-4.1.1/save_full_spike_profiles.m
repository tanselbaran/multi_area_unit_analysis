%%To use this script, make sure to have enabled the exportResults option in
%%the parameters file, and then save the spikesFilt variable from inside the 
%res structure in a separate .mat file Then load the _res.mat file for loading the results of the
%%manual curation in the workspace.

spikesFilt = res.spikesFilt;
clusterNotes = res.clusterNotes;

%determine which sites are selected by JRclust for the representation of
%each cluster
cluster_sites_full = zeros(size(res.meanWfGlobal,3), size(res.meanWfGlobal,2));
for i=1:size(res.meanWfGlobal,3)
   cluster_sites_full(i,:) = ismember(res.meanWfGlobal(1,:,i), res.meanWfLocal(1,:,i));
end

singleUnitClusters = [];

for i=1:size(clusterNotes,1)
    if strcmp(clusterNotes{i},'single')
        singleUnitClusters(end+1)=i;
    end
end

%eliminate the unsorted and multiunit spikes 
sortedSpikes = find(ismember(res.spikeClusters,singleUnitClusters));
spikeTimesSorted = res.spikeTimes(sortedSpikes);
sortedSpikeClusters = res.spikeClusters(sortedSpikes);

%populating the array with size (N_samples x N_channels x N_sorted_spikes)
%with the filtered spikes from a limited number of channels isolated by
%JRClust
spikesFiltFull = zeros(size(res.meanWfGlobal,1),size(res.meanWfGlobal,2),size(sortedSpikes,1),'int16');
for i = 1:size(sortedSpikes,1)
    idx = find(cluster_sites_full(sortedSpikeClusters(i),:) == 1);
    spikesFiltFull(:,idx,i) = spikesFilt(:,:,sortedSpikes(i));
end

%saving the waveforms, cluster identities and times of the sorted spikes in
%the file "sorted_spike_data.mat"
save('sorted_spike_data.mat','sortedSpikes','cluster_sites_full', 'clusterNotes', 'spikeTimesSorted','sortedSpikeClusters','spikesFiltFull','-v7.3')
clear