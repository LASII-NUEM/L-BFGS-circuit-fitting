import pandas as pd
import csv
import numpy as np
import os
from framework import data_types

def read(filename:str, n_samples:int, electrode="flange", acquisition_mode="freq",testmode = "sequential", aggregate=None, timezone=-3):
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
        raise FileNotFoundError(f'[file_lcr] Filename {filename} does not exist!')

    #validate n_samples
    if not (n_samples>0):
        raise ValueError(f'[file_lcr] n_samples = {n_samples}, must be > 0!')

    #validate electrode
    electrode = electrode.lower() #convert to lowercase
    valid_electrode = ['flange', 'cell', 'custom'] #list of valid sweep types
    if electrode not in valid_electrode:
        raise ValueError(f'[file_lcr] electrode = {electrode} not implemented! Try: {valid_electrode}')

    #validate acquisition_mode
    acquisition_mode = acquisition_mode.lower()  # convert to lowercase
    valid_acqs = ['freq', 'spectrum']  # list of valid sweep types
    if acquisition_mode not in valid_acqs:
        raise ValueError(f'[file_lcr] acquisition_mode = {acquisition_mode} not implemented! Try: {valid_acqs}')

    #infer the swept frequencies from the file header
    with open(filename, 'r') as f:
        reader = csv.DictReader(f) #read only the header
        header_data = reader.fieldnames
    f.close() #close the file

    #process the raw data output from the PHOBOS acquisition system into a custom data structure
    raw_data = pd.read_csv(filename).to_numpy() #process the raw data output from the PHOBOS acquisition system

    if electrode == "flange":
        if acquisition_mode == 'freq':
            swept_freqs = [float(freq.replace(" ", "").replace("Cp", "").replace("Z", "")) for freq in header_data if ('Cp' in freq or 'Z' in freq)]
            swept_freqs = np.array(swept_freqs) #convert to numpy array
            data = data_types.FlangeData(raw_data, swept_freqs, n_samples, aggregate=aggregate, timezone=timezone)
        elif acquisition_mode == 'spectrum':
            swept_freqs = [float(freq.replace(" ", "").replace("Cp", "").replace("Z", "")) for freq in header_data if ('Cp' in freq or 'Z' in freq)]
            swept_freqs = np.array(swept_freqs)  # convert to numpy array
            data = data_types.SpectroscopyData(raw_data, swept_freqs, n_samples, electrode=electrode,testmode = testmode, aggregate=aggregate)

    elif electrode == "cell":
        if acquisition_mode == 'freq':
            swept_freqs = [float(freq.replace(" ", "").replace("Cp", "").replace("Z", "")) for freq in header_data if ('Cp' in freq or 'Z' in freq)]
            swept_freqs = np.array(swept_freqs) #convert to numpy array
            data = data_types.CommercialCellData(raw_data, swept_freqs, n_samples, aggregate=aggregate)
        elif acquisition_mode == 'spectrum':
            swept_freqs = [float(freq.replace(" ", "").replace("Cp", "").replace("Z", "")) for freq in header_data if ('Cp' in freq or 'Z' in freq)]
            swept_freqs = np.array(swept_freqs) #convert to numpy array
            data = data_types.SpectroscopyData(raw_data, swept_freqs, n_samples, electrode=electrode, testmode = testmode,aggregate=aggregate)

    elif electrode == "custom":
        data = None #TODO: generate custom-based files

    return data