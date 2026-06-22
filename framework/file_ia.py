import pandas as pd
import numpy as np
import os
from framework import data_types

def read(filename:str):
    '''
    :param filename: path where the .xls is stored
    '''

    #check if the filename exists
    if not os.path.isfile(filename):
        raise FileNotFoundError(f'[file_lcr] Filename {filename} does not exist!')

    xls = pd.ExcelFile(filename)
    sheet_names = xls.sheet_names

    row_start = 9
    row_end = 210

    n_freq = row_end - row_start

    raw_data = np.zeros((2, n_freq), dtype=float)

    data = {}
    for i in range(2, len(sheet_names)):
        df = pd.read_excel(xls, sheet_name=sheet_names[i], usecols=[2, 3, 5], skiprows=row_start,
                           nrows=row_end - row_start, dtype="float64")

        swept_freqs = pd.to_numeric(df.iloc[:, 0], errors='coerce') * 1e-14
        df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='coerce') * 1e-6
        df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce') * 1e-6

        df.rename(columns={df.columns[1]: 'CP', df.columns[2]: 'RP'}, inplace=True)

        Cp = df["CP"].to_numpy()
        Rp = df["RP"].to_numpy()

        raw_data[0, :] = Cp
        raw_data[1, :] = Rp

        swept_freqs = np.array(swept_freqs) #convert to numpy array
        data[sheet_names[i].lower()] = data_types.SpectroscopyData(raw_data, swept_freqs, hardware="ia")

    return data
