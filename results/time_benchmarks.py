import os
import sys

#ensure framework is identified when running in terminal
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from framework import batch_utils, fitting_utils, file_ucsd
from scipy.io import loadmat
import numpy as np

def fit_methane(filename, model):
    ucsd_obj = file_ucsd.read(filename)
    circuits = {"Constable2020": {"guess": np.array([1, 1, 1, 0.5]),
                                  "scale_BFGS": np.array([1e3, 1e2, 1e-3, 1]),
                                  "scale_NLLS": np.array([1e2, 1e2, 1e-5, 1]),
                                  "scale_NelderMead": np.array([1e2, 1e2, 1e-5, 1]),
                                  "scale_PSO": np.array([1e2, 1e2, 1e-5, 1])},
                "Lu2019_2CPE": {"guess": np.array([1, 1, 0.5, 1, 1, 0.5]),
                                "scale_BFGS": np.array([1e3, 1e-3, 1, 1e3, 1e-5, 1]),
                                "scale_NLLS": np.array([1e3, 1e-3, 1, 1e3, 1e-5, 1]),
                                "scale_NelderMead": np.array([1e4, 1e-3, 1, 1e3, 1e-5, 1]),
                                "scale_PSO": np.array([1e3, 1e-3, 1, 1e3, 1e-5, 1])},
                "Lu2019_CPECP": {"guess": np.array([1, 1, 0.5, 1, 1]),
                                 "scale_BFGS": np.array([1e7, 1e-3, 1, 1e5, 1e-7]),
                                 "scale_NLLS": np.array([1e7, 1e-3, 1, 1e5, 1e-7]),
                                 "scale_NelderMead": np.array([1e7, 1e-3, 1, 1e5, 1e-7]),
                                 "scale_PSO": np.array([1e7, 1e-3, 1, 1e5, 1e-7])},
                "Roberts1993": {"guess": np.array([1, 1, 1, 1]),
                                "scale_BFGS": np.array([1e8, 1e-11, 1e5, 1e-12]),
                                "scale_NLLS": np.array([1e7, 1e-11, 1e5, 1e-12]),
                                "scale_NelderMead": np.array([1e8, 1e-11, 1e7, 1e-12]),
                                "scale_PSO": np.array([1e7, 1e-11, 1e5, 1e-12])}
                } #list of circuits to attempt fitting the data

    #minimizer arguments
    init_guess_BFGS = circuits[model]["guess"]
    scaling_array_BFGS = circuits[model]["scale_BFGS"]
    init_guess_NLLS = circuits[model]["guess"]
    scaling_array_NLLS = circuits[model]["scale_NLLS"]
    init_guess_simplex = circuits[model]["guess"]
    scaling_array_simplex = circuits[model]["scale_NelderMead"]

    #impedance fitting w/ BFGS
    timestamps = np.zeros(shape=(3,1))
    fit_obj = fitting_utils.EquivalentCircuit(model, [ucsd_obj.z_meas_real, ucsd_obj.z_meas_imag], ucsd_obj.freqs) #equivalent circuit object
    fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=False)
    timestamps[0,0] = fit_params_BFGS.t_elapsed
    fit_params_NLLS = fit_obj.fit_circuit(init_guess_NLLS, scaling_array_NLLS, method="NLLS", tol=1e-8, verbose=False)
    timestamps[1,0] = fit_params_NLLS.t_elapsed
    fit_params_simplex = fit_obj.fit_circuit(init_guess_simplex, scaling_array_simplex, method="Nelder-Mead", tol=1e-6, verbose=False)
    timestamps[2,0] = fit_params_simplex.t_elapsed

    return timestamps

def fit_thf(filename, model):
    data = loadmat(filename)
    freqs = data['f'].flatten()
    freq_mask = freqs >= 100
    freqs = freqs[freq_mask]
    omegas = 2*np.pi*freqs
    Rp = data['rpcolunas'][freq_mask, :]
    Cp = data['cpcolunas'][freq_mask, :]
    z_line_medium = Rp/(1+(omegas[:, np.newaxis]*Cp*Rp)**2) #real part of the impedance
    z_line_medium = z_line_medium[:, -1]
    z_2line_medium = (omegas[:, np.newaxis]*Cp*(Rp ** 2))/(1+(omegas[:, np.newaxis]*Cp*Rp)**2) #imaginary part of the impedance
    z_2line_medium = z_2line_medium[:, -1]
    Z_meas = z_line_medium-1j*z_2line_medium
    Z_meas = Z_meas.astype('complex')
    circuits = {"Longo2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 0.5, 1]),
                              "scale_BFGS": np.array([1e5, 1e-4, 1e-2, 1e-11, 1e5, 1, 1, 1e-8]),
                              "scale_NLLS": np.array([1e5, 1e-4, 1e-2, 1e-11, 1e5, 1, 1, 1e-8]),
                              "scale_NelderMead": np.array([1e5, 1e-3, 1e-2, 1e-11, 1e5, 1, 1, 1e-8]),
                              "scale_PSO": np.array([1e5, 1e-4, 1e-3, 1e-12, 1e5, 1, 1, 1e-9])}} #list of circuits to attempt fitting the data

    # minimizer arguments
    init_guess_BFGS = circuits[model]["guess"]
    scaling_array_BFGS = circuits[model]["scale_BFGS"]
    init_guess_NLLS = circuits[model]["guess"]
    scaling_array_NLLS = circuits[model]["scale_NLLS"]
    init_guess_simplex = circuits[model]["guess"]
    scaling_array_simplex = circuits[model]["scale_NelderMead"]

    #impedance fitting w/ BFGS
    timestamps = np.zeros(shape=(3, 1))
    fit_obj = fitting_utils.EquivalentCircuit(model, [Z_meas.real, -Z_meas.imag], freqs)  # equivalent circuit object
    fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=False)
    timestamps[0, 0] = fit_params_BFGS.t_elapsed
    fit_params_NLLS = fit_obj.fit_circuit(init_guess_NLLS, scaling_array_NLLS, method="NLLS", tol=1e-8, verbose=False)
    timestamps[1, 0] = fit_params_NLLS.t_elapsed
    fit_params_simplex = fit_obj.fit_circuit(init_guess_simplex, scaling_array_simplex, method="Nelder-Mead", tol=1e-6, verbose=False)
    timestamps[2, 0] = fit_params_simplex.t_elapsed

    return timestamps

def fit_ice(filepath, model):
    reject_freqs = 100 #reject all frequencies up to the given argument
    dirs = os.listdir(filepath) #list all files in the path
    sample_name = 'distilled'

    #fit water circuits
    circuits = {"distilled": {"Lu2019_2CPE": {"guess": np.array([1, 1, 0.5, 1, 1, 0.5]),
                                              "scale_BFGS": np.array([1e3, 1e-7, 1, 1e3, 1e-3, 1]),
                                              "scale_NLLS": np.array([1e3, 1e-7, 1, 1e3, 1e-3, 1]),
                                              "scale_NelderMead": np.array([1e4, 1e-4, 1, 1e3, 1e-5, 1]),
                                              "scale_PSO": np.array([1e3, 1e-7, 1, 1e3, 1e-4, 1])}}}

    #batch process PHOBOS spectroscopy acquisition
    curr_dir = [cand_dir for cand_dir in dirs if sample_name in cand_dir][0]
    curr_dir = os.path.join(filepath, curr_dir)
    curr_obj = batch_utils.BatchData(curr_dir, topology={}, freq_threshold=reject_freqs, n_samples_freq=1,
                                     n_samples_spectrum=3,
                                     electrode="cell", hardware='LCR', aggregate=np.mean, fit=False)

    #minimizer arguments
    init_guess_BFGS = circuits[sample_name][model]["guess"]
    scaling_array_BFGS = circuits[sample_name][model]["scale_BFGS"]
    init_guess_NLLS = circuits[sample_name][model]["guess"]
    scaling_array_NLLS = circuits[sample_name][model]["scale_NLLS"]
    init_guess_simplex = circuits[sample_name][model]["guess"]
    scaling_array_simplex = circuits[sample_name][model]["scale_NelderMead"]

    #impedance fitting w/ BFGS
    timestamps = np.zeros(shape=(3, 1))
    fit_obj = fitting_utils.EquivalentCircuit(model, [curr_obj.spec_ice_obj.z_real, curr_obj.spec_ice_obj.z_imag],
                                              curr_obj.spec_ice_obj.media_obj.freqs)  # equivalent circuit object
    fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=False)
    timestamps[0, 0] = fit_params_BFGS.t_elapsed
    fit_params_NLLS = fit_obj.fit_circuit(init_guess_NLLS, scaling_array_NLLS, method="NLLS", tol=1e-8, verbose=False)
    timestamps[1, 0] = fit_params_NLLS.t_elapsed
    fit_params_simplex = fit_obj.fit_circuit(init_guess_simplex, scaling_array_simplex, method="Nelder-Mead", tol=1e-6, verbose=False)
    timestamps[2, 0] = fit_params_simplex.t_elapsed

    return timestamps

#filepaths with the spectra
ch4_file = os.path.join('../../data/methane_hydrate/', f'CH4_GwINSTEK_LCR_spectrum.txt')
ch4_sand_file = os.path.join('../../data/methane_hydrate/', f'CH4_sand_GwINSTEK_LCR_spectrum.txt')
thf_file = os.path.join('../../data/tetrahydrofuran_hydrate/', f'thf_4294A_ia')
ice_files = '../data/ice' #path where all test files are stored

#N independent runs to compute average computing time
n_runs = 100
t_ch4 = []
t_ch4_sand = []
t_thf = []
t_ice = []
for i in range(0,n_runs):
    print(f'[time_benchmarks] run = {i+1}/{n_runs}')
    t_ch4.append(fit_methane(ch4_file, 'Constable2020'))
    t_ch4_sand.append(fit_methane(ch4_sand_file, 'Lu2019_2CPE'))
    t_thf.append(fit_thf(thf_file, 'Longo2020'))
    t_ice.append(fit_ice(ice_files, 'Lu2019_2CPE'))

#convert to array
t_ch4 = np.array(t_ch4)[:,:,0]
t_ch4_sand = np.array(t_ch4_sand)[:,:,0]
t_thf = np.array(t_thf)[:,:,0]
t_ice = np.array(t_ice)[:,:,0]

#compute the average for each algorithm
print(f'[CH4] {np.mean(t_ch4, axis=0)}')
print(f'[CH4 (sand)] {np.mean(t_ch4_sand, axis=0)}')
print(f'[THF] {np.mean(t_thf, axis=0)}')
print(f'[ice] {np.mean(t_ice, axis=0)}')