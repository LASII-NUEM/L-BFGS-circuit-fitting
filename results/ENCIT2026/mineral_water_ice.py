import os
import sys

#ensure framework is identified when running in terminal
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from framework import batch_utils, fitting_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif'] #font configuration (same as paper)

#input arguments
sample_name = 'mineral'
data_path = '../../data/ice/' #path where all methane-hydrate-related files are stored
reject_freqs = 100 #reject all frequencies up to the given argument

#load the data into a data_types.SpectroscopyData object
dirs = os.listdir(data_path) #list all files in the data path
sample_dir = [cand_dir for cand_dir in dirs if sample_name in cand_dir][0] #sample-name-related directory
sample_dir = os.path.join(data_path, sample_dir) #relative path to where spectroscopy data from the sample is stored
sample_obj = batch_utils.BatchData(sample_dir, topology={}, freq_threshold=reject_freqs, n_samples_freq=1, n_samples_spectrum=3,
                                   electrode="cell", hardware='LCR', aggregate=np.mean, fit=False)
ice_obj = sample_obj.spec_ice_obj #ice-related data
freqs = sample_obj.freerun_obj.freqs #frequencies of the sweep

#Linear Kramers-Kronig with overfitting criterion
linKK = fitting_utils.LinearKramersKronig([ice_obj.z_real, ice_obj.z_imag], freqs, c=0.5, max_iter=100, add_capacitor=True, verbose=False)
print(f'[Linear Kramers-Kronig] M = {linKK.fit_components}')
print(f'[Linear Kramers-Kronig] Δ real = {np.mean(linKK.fit_residues_real)}%')
print(f'[Linear Kramers-Kronig] Δ imag = {np.mean(linKK.fit_residues_imag)}%')

circuits= {"Longo2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 0.5, 1]),
                          "scale_BFGS": np.array([1e5, 1e-6, 1e7, 1e-2, 1e2, 1e-5, 1, 1]),
                          "scale_NLLS": np.array([1e5, 1e-7, 1e7, 1e-2, 1e2, 1e-5, 1, 1])},
            "Zurich2021": {"guess": np.array([1, 1, 0.5, 1, 1, 1, 1]),
                          "scale_BFGS": np.array([1e3, 1e-9, 1, 1e11, 1e-4, 1e5, 1e-10]),
                          "scale_NLLS": np.array([1e3, 1e-9, 1, 1e11, 1e-4, 1e5, 1e-10])},
            "Roberts1993": {"guess": np.array([1, 1, 1, 1]),
                            "scale_BFGS": np.array([1e3, 1e-9, 1e4, 1e-8]),
                            "scale_NLLS": np.array([1e3, 1e-9, 1e4, 1e-8])},
            "Lu2019_2CPE": {"guess": np.array([1, 1, 0.5, 1, 1, 0.5]),
                           "scale_BFGS": np.array([1e7, 1e-2, 1, 1e8, 1e-1, 1]),
                           "scale_NLLS": np.array([1e4, 1e-3, 1, 1e8, 1e-1, 1])}}

#fit for all selected models
fitted_data = {} #dictionary to store all processed models
for model in circuits.keys():
    #algorithm-specific arguments
    init_guess_BFGS = circuits[model]["guess"]
    scaling_array_BFGS = circuits[model]["scale_BFGS"]
    init_guess_NLLS = circuits[model]["guess"]
    scaling_array_NLLS = circuits[model]["scale_NLLS"]

    #run electric circuit fitting with all the screened algorithms
    fit_obj = fitting_utils.EquivalentCircuit(model, [ice_obj.z_real, ice_obj.z_imag], freqs) #equivalent circuit object
    fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=True) #L-BFGS fit
    fit_params_NLLS = fit_obj.fit_circuit(init_guess_NLLS, scaling_array_NLLS, method="NLLS", tol=1e-8, verbose=True) #NLLS fit
    fitted_data[model] = {"BFGS": fit_params_BFGS, "NLLS": fit_params_NLLS} #store fitted data

    #Nyquist plots with measured and fitted data
    fig, ax = plt.subplots()
    leg = []
    ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue", s=20)
    leg.append('measured')
    ax.plot(fit_params_BFGS.opt_fit.real, -fit_params_BFGS.opt_fit.imag, color="tab:orange", linewidth=2.)
    leg.append(f'L-BFGS: χ² = {fit_params_BFGS.chi_square}')
    ax.plot(fit_params_NLLS.opt_fit.real, -fit_params_NLLS.opt_fit.imag, color="tab:green", linewidth=.75)
    leg.append(f'NLLS: χ² = {fit_params_NLLS.chi_square}')
    plt.xlabel("Z'")
    plt.ylabel("Z''")
    plt.title(f"Model = {model}")
    plt.legend(leg)
    plt.grid()
    plt.show()

    #reduced size
    x_size, y_size = plt.gcf().get_size_inches()
    size_ratio = .5
    plt.figure(figsize=(x_size*size_ratio, y_size*size_ratio))
    plt.semilogy(np.arange(1,len(fit_obj.costs)+1,1), fit_obj.costs)
    plt.title(f"Model = {model}")
    plt.xlabel('Iteration')
    plt.ylabel('SSE')
    plt.grid()
    plt.tight_layout()
    plt.show()