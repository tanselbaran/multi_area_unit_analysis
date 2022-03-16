"""
author: Tansel Baran Yasar

This script is used for downsampling the raw multi-channel electrophysiology data stored in a binary .dat file with int16 format, and storing the downsampled LFP data in another binary .dat file also with int16 format. The script assumes the name of the data file is amplifier.dat. 
"""

import math
import numpy as np
from scipy.signal import filtfilt, butter, decimate

def read_time_dat_file(filepath, sample_rate):
    """
    This function reads the time array from a time.dat file created by Intan software for a recording session,
    and returns the time as a numpy array.

    Inputs:
        filepath: The path to the time.dat file to be read.
        sample_rate: Sampling rate for the recording of interest.

    Outputs:
        time_file: Numpy array containing the time array (in s)
    """

    with open(filepath, 'rb') as fid:
        raw_array = np.fromfile(fid, np.int32)
    time_file = raw_array / float(sample_rate) #converting from int32 to seconds
    return time_file

###########SET PARAMETERS HERE#################
sample_rate = 20000
ds_sample_rate = 2000
ds_factor = int(sample_rate / ds_sample_rate)
num_channels = 256
chunk_size = 60
raw_data_dir = '/home/baran/Desktop/rTBY34/3_freely_behav_220307_130622'
bit_to_uV = 0.195
##############################################33

raw_data_file = raw_data_dir + '/amplifier.dat'
ds_data_file = raw_data_dir + '/amplifier_ds.dat'
time_file = raw_data_dir + '/time.dat'

#Calculating the number of dat chunks
time = read_time_dat_file(time_file,sample_rate)
recording_len = time[-1]
num_chunks = math.ceil(recording_len / chunk_size)

f = open(ds_data_file,'wb')

for chunk in range(num_chunks):
    print("Processing chunk: "+str(chunk)+" out of "+str(num_chunks))

    #Calculating the starting point of the chunk in the .dat file in bytes
    start_idx = chunk * chunk_size * sample_rate * num_channels
    start_offset = int(start_idx*2)

    #Reading the binary data of the chunk.
    if chunk == (num_chunks - 1):
        data = np.fromfile(raw_data_file, 'int16', offset = start_offset)
    else:
        end_idx  = (chunk+1) * chunk_size * sample_rate * num_channels
        data = np.fromfile(raw_data_file, 'int16', end_idx-start_idx, offset=start_offset)

    #Reshaping the data chunk, decimating it, and writing it into the output file.
    data = np.reshape(data,(num_channels,int(len(data)/num_channels)),'F') * bit_to_uV
    data_ds = decimate(data,ds_factor)
    data_ds = data_ds.flatten('F')
    data_ds = data_ds.astype('int16')
    data_ds.tofile(f)

f.close()
