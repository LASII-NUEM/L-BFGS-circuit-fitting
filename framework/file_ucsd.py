import numpy as np
import os
from framework import data_types

def read(filename:str):
    '''
    :param filename: path where the .csv is stored
    :param n_samples: samples per pair swept
    :param electrode: which hardware was used to acquire the signals
    :param acquisition_mode: how the data is expected to be organized ('flange' for 10-mode sweep or 'spectrum' for full spectroscopy)
    :param aggregate: how to organize the data for each mode (None as default)
    :param timezone: timezone to convert unix timestamp to human timestamp
    '''

    #check if the filename exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'[file_ucsd] Filename {filename} does not exist!')

    #read the text files line by line and append as float
    raw_data = [] #list to append the readings
    with open(filename) as ucsdfile:
        ucsdfile.readline() #skip test number header
        for line in ucsdfile:
            line_att = line.split() #split readings
            if not line_att:
                continue
            raw_data.append(line_att) #append to the raw data list

    raw_data = np.array(raw_data).astype(float) #convert to numpy array and set the data as float
    raw_data = raw_data[::-1] #reverse the frequency order (smaller frequencies first)
    data = data_types.UCSData(raw_data) #generate the objet of the UCSData class

    return data
