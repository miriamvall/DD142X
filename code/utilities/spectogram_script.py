from os import walk, mkdir
from h5py import File

import numpy as np

def directory_script(
    directoryIn = "../_data/matlabData",
    directoryOut = "../_data/amplitudeSpectograms",
    Fs = 16000.,
    epoch_size = 2 ** 11,
    fft_n = 2 ** 14,
    lo = 0,
    hi = 200,
    # Memory control variable
    memory_rows_fft_max = 400
):
    for _, _, fileList in walk(directoryIn):
        for filename in fileList:
            subdirOut = filename.replace(".mat", "")
            file_script(
                f"{directoryIn}/{filename}", 
                f"{directoryOut}/{subdirOut}"
            )

def file_script(
    fileIn,
    dirOut, 
    patterns = ["str_lfp", "gp_lfp"],
    Fs = 16000.,
    epoch_size = 2 ** 11,
    fft_n = 2 ** 14,
    lo = 0,
    hi = 300,
    # Memory control variable
    memory_rows_fft_max = 400
):
    try:
        mkdir(dirOut)
    except:
        pass

    def pattern_match(key):
        for pattern in patterns:
            if pattern in key:
                return True
        return False

    freqs = np.fft.fftfreq(fft_n, 1. / Fs)
    loidx = np.where(freqs > lo)[0][0]
    hiidx = np.where(freqs > hi)[0][0]
    np.save(f"{dirOut}/freqs", freqs[loidx : hiidx])

    with File(fileIn, "r") as mlData:

        for channel, values in [
            (
                key, 
                np.array(dict_["values"])
            ) 
            for key, dict_ in mlData.items()
            if pattern_match(key)
        ]:
        
            max_idx = epoch_size * int(values.shape[1] / epoch_size)
            values = values[ : , 0 : max_idx].reshape((-1, epoch_size))
            ffts = np.abs(
                np.fft.fft(values, fft_n)[ : , loidx : hiidx]
            )
            np.save(f"{dirOut}/{channel}", ffts)

def load_all(directoryIn = "../_data/amplitudeSpectograms"):
    freqs = ""
    data = dict()
    for _, dirs, _ in walk(directoryIn):
        for session in dirs:
            for _, _, channels in walk(f"{directoryIn}/{session}"):
                
                if type(freqs) == str:
                    freqs = np.load(f"{directoryIn}/{session}/freqs.npy")

                data[session] = {
                    channel.replace(".npy", "") : np.load(f"{directoryIn}/{session}/{channel}")
                    for channel in channels
                    if channel != "freqs.npy"
                }

        break
    return freqs, data

def epochs_as_rows(
    data_dict
):
    return np.concatenate([
        np.concatenate([
            epochs 
            for channelName, epochs in channelData.items()
        ], axis = 0)
        for sessionName, channelData in data_dict.items()
    ], axis = 0)

###############################
# pwd
# <...>/DD142X/code/utilities
# Uncomment next row and run program to generate all data
# directory_script()
###############################

###############################
# USAGE:
# frequency_vector, data_dictionary = load_all()
# epochs_as_rows = epochs_as_rows(data_dictionary)
#   
# epochs.shape (with full dataset, default params) == (367629, 307)
###############################