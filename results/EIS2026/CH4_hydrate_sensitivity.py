import os
from framework import fitting_utils, file_ucsd, sensitivity_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif'] #font configuration (same as paper)

#input arguments
sample_file = 'CH4_GwINSTEK_LCR' #CH4 methane synthesized from salt-water
data_path = '../../data/methane_hydrate/' #path where all methane-hydrate-related files are stored
model = "Constable2020" #which ECM will be used to fit the data

#initial arguments
circuits = {"Constable2020": {"guess": np.array([1, 1, 1, 0.5]),
                              "scale_BFGS": np.array([1e3, 1e2, 1e-3, 1])},
            "Constable2020_LogCPE": {"guess": np.array([1, 1, 1, 0.5]),
                              "scale_BFGS": np.array([1e3, 1e2, -3, 1])}} #dictionary with initial guess and scaling arrays

#MonteCarlo bias analysis
n_spectra = 1e4 #number of simulations
verbose = False
proc_parallel = True #compute the simulations in parallel
cores = -1 #number of cors (-1 utilizes all)

if __name__ == '__main__':
    #load the data into a data_types.UCSData object
    filename = os.path.join(data_path, f'{sample_file}_spectrum.txt') #relative path to the CH4 hydrate data
    ucsd_obj = file_ucsd.read(filename)
    os.path.join('../data/methane_hydrate/', f'CH4_GwINSTEK_LCR_spectrum.txt')

    #Linear Kramers-Kronig with overfitting criterion
    linKK = fitting_utils.LinearKramersKronig([ucsd_obj.z_meas_real, ucsd_obj.z_meas_imag], ucsd_obj.freqs, c=0.5, max_iter=100, add_capacitor=True, verbose=False)
    print(f'[Linear Kramers-Kronig] M = {linKK.fit_components}')
    print(f'[Linear Kramers-Kronig] Δ real = {np.mean(linKK.fit_residues_real)}%')
    print(f'[Linear Kramers-Kronig] Δ imag = {np.mean(linKK.fit_residues_imag)}%')

    #algorithm-specific arguments
    init_guess_BFGS = circuits[model]["guess"]
    scaling_array_BFGS = circuits[model]["scale_BFGS"]

    #run electric circuit fitting with all the screened algorithms
    fit_obj = fitting_utils.EquivalentCircuit(model, [ucsd_obj.z_meas_real, ucsd_obj.z_meas_imag], ucsd_obj.freqs) #equivalent circuit object
    fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=True) #L-BFGS fit

    #load fitted data from the database
    fit_filename = os.path.join(data_path, f'{sample_file}_fitted.txt') #relative path to the CH4 hydrate fitted data
    ucsd_fit_obj = file_ucsd.read(fit_filename) #load the fitted data into a data_types.UCSData object
    ucsd_z_meas = ucsd_obj.z_meas_real-1j*ucsd_obj.z_meas_imag #complex impedance
    ucsd_z_meas = ucsd_z_meas.astype('complex')
    inter_freqs = ucsd_fit_obj.freqs #interpolated frequencies
    model = model.lower() #convert to lower case
    ecm_circuit = fitting_utils.function_handlers[model]["function_ptr"] #pointer to the ECM function given the model
    interp_BFGS = ecm_circuit(fit_params_BFGS.opt_params, [2*np.pi*inter_freqs, scaling_array_BFGS]) #interpolated L-BFGS output
    interp_BFGS = interp_BFGS.astype('complex')

    #local sensitivity analysis
    theta = fit_params_BFGS.opt_params
    eps = np.sqrt(np.finfo(float).eps) #machine epsilon
    sigma_sqr = np.sqrt(fit_params_BFGS.opt_cost/(2*len(ucsd_obj.freqs)-len(fit_params_BFGS.opt_params)))
    sens_obj = sensitivity_utils.LocalIdentifiabilityAnalysis(sigma_sqr, ucsd_obj.freqs, theta, model, eps=eps,
                                                 fit_freqs=ucsd_obj.freqs, scaling_array=scaling_array_BFGS, verbose=True)
    S_theta = sens_obj.S_theta

    #MonteCarlo bias analysis
    Z_simu = fitting_utils.function_handlers[model.lower()]["function_ptr"](theta, args=[2*np.pi*ucsd_obj.freqs, scaling_array_BFGS])
    Z_simu = Z_simu.astype("complex")
    mc_obj = sensitivity_utils.MonteCarloFit(model, n_spectra, sigma_sqr, theta, scaling_array_BFGS, ucsd_obj.freqs) #Monte Carlo fit object
    Z_fit, theta_fit = mc_obj.fit_simulations(init_guess_BFGS, scaling_array_BFGS, parallel=proc_parallel, n_cores=cores,
                                              method='BFGS', verbose=verbose, plot=False) #fit the simulations
    fit_mean, fit_var, fit_std = mc_obj.compute_statistics(theta_fit) #compute the fit statistics
    relat_bias = 100*np.abs(fit_mean-(theta*scaling_array_BFGS))/(theta*scaling_array_BFGS)
    params = fitting_utils.function_handlers[model.lower()]["fit_params"]

    print(f'[IdentifiabilityAnalysis] Uncertainty:')
    for name, val_real, val_phys, err_abs, rel_bias, err_pct in zip(params, theta*scaling_array_BFGS,
                                                fit_mean, fit_std, relat_bias, sens_obj.rel_std):
        print(f"{name} | {val_real} | {val_phys} | ± {err_abs} | ± {rel_bias} | {err_pct}")

    #Nyquist plots with measured and fitted data
    fig, ax = plt.subplots()
    leg = []
    ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue", s=20)
    leg.append('measured')
    ax.plot(interp_BFGS.real, -interp_BFGS.imag, color="tab:orange", linewidth=2.)
    leg.append(f'L-BFGS: χ² = {fit_params_BFGS.chi_square}')
    plt.xlabel("ℜ(Z) [Ω]")
    plt.ylabel("-ℑ(Z) [Ω]")
    plt.legend(leg)
    plt.grid()
    plt.show()

    #plot the real part of the computed sensitivity
    plt.figure()
    plot_params = fitting_utils.function_handlers[model]["fit_params"]
    for i in range(len(plot_params)):
        plt.semilogx(ucsd_obj.freqs, np.real(S_theta[:,i]), label=plot_params[i])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Re(S^Z_theta)')
    plt.legend()
    plt.grid()
    plt.show()

    #plot the imaginary part of the computed sensitivity
    plt.figure()
    plot_params = fitting_utils.function_handlers[model]["fit_params"]
    for i in range(len(plot_params)):
        plt.semilogx(ucsd_obj.freqs, np.imag(S_theta[:,i]), label=plot_params[i])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Im(S^Z_theta)')
    plt.legend()
    plt.grid()
    plt.show()
