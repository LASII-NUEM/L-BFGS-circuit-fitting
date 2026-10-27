import os
from framework import fitting_utils, sensitivity_utils
from scipy.io import loadmat
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif'] #font configuration (same as paper)

#input arguments
filename = '../../data/tetrahydrofuran_hydrate/thf_4294A_ia.mat' #path where all test files are stored
model = "LongoRed2020" #which ECM will be used to fit the data
reject_freqs = 100 #reject all frequencies up to the given argument

#initial arguments
circuits = {"LongoAdp2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 0.5]),
                             "scale_BFGS": np.array([1e5, 1e-7, 1e-4, 1e-9, 1e6, 1e-8, 1])},
            "LongoRed2020": {"guess": np.array([1, 1, 1, 0.5]),
                             "scale_BFGS": np.array([1e-9, 1e6, 1e-8, 1])},
            "LongoLog2020": {"guess": np.array([1, 1, 1, 1, 1, 1, 0.5]),
                             "scale_BFGS": np.array([1e5, 1e-7, 1e-3, 1e-9, 1e5, -8, 1])}} #dictionary with initial guess and scaling arrays

#MonteCarlo bias analysis
n_spectra = 1e1 #number of MC simulations
verbose = False
proc_parallel = True #compute the simulations in parallel
cores = -1 #number of cores (-1 utilizes all)

if __name__ == '__main__':
    #load the data and compute the complex impedance
    data = loadmat(filename) #load .MAT file
    freqs = data['f'].flatten() #frequencies array
    freq_mask = freqs>=reject_freqs #mask to reject frequencies below 100 Hz due to system noise
    freqs = freqs[freq_mask] #mask the frequencies
    omegas = 2*np.pi*freqs #Hz to rad/s
    Rp = data['rpcolunas'][freq_mask,:] #mask the resistance readings
    Cp = data['cpcolunas'][freq_mask,:] #mask the capacitance readings
    z_line_medium = Rp/(1+(omegas[:,np.newaxis]*Cp*Rp)**2) #real part of the impedance
    z_line_medium = z_line_medium[:,-1]
    z_2line_medium = (omegas[:,np.newaxis]*Cp*(Rp**2))/(1+(omegas[:,np.newaxis]*Cp*Rp)**2) #imaginary part of the impedance
    z_2line_medium = z_2line_medium[:,-1]
    Z_meas = z_line_medium - 1j*z_2line_medium #complex impedance
    Z_meas = Z_meas.astype('complex')

    #Linear Kramers-Kronig with overfitting criterion
    linKK = fitting_utils.LinearKramersKronig([Z_meas.real, -Z_meas.imag], freqs, c=0.5, max_iter=100, add_capacitor=True, verbose=False)
    print(f'[Linear Kramers-Kronig] M = {linKK.fit_components}')
    print(f'[Linear Kramers-Kronig] Δ real = {np.mean(linKK.fit_residues_real)}%')
    print(f'[Linear Kramers-Kronig] Δ imag = {np.mean(linKK.fit_residues_imag)}%')

    #algorithm-specific arguments
    init_guess_BFGS = circuits[model]["guess"]
    scaling_array_BFGS = circuits[model]["scale_BFGS"]

    #run electric circuit fitting with all the screened algorithms
    fit_obj = fitting_utils.EquivalentCircuit(model, [Z_meas.real, -Z_meas.imag], freqs) #equivalent circuit object
    fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=True) #L-BFGS fit

    #local sensitivity analysis
    theta = fit_params_BFGS.opt_params
    eps = np.sqrt(np.finfo(float).eps) #machine epsilon
    sigma_sqr = np.sqrt(fit_params_BFGS.opt_cost/(2*len(freqs)-len(fit_params_BFGS.opt_params)))
    sens_obj = sensitivity_utils.LocalIdentifiabilityAnalysis(sigma_sqr, freqs, theta, model.lower(), eps=eps,
                                                 fit_freqs=freqs, scaling_array=scaling_array_BFGS, verbose=True)
    S_theta = sens_obj.S_theta

    #MonteCarlo bias analysis
    Z_simu = fitting_utils.function_handlers[model.lower()]["function_ptr"](theta, args=[2*np.pi*freqs, scaling_array_BFGS])
    Z_simu = Z_simu.astype("complex")
    mc_obj = sensitivity_utils.MonteCarloFit(model, n_spectra, sigma_sqr, theta, scaling_array_BFGS, freqs) #Monte Carlo fit object
    Z_fit, theta_fit = mc_obj.fit_simulations(init_guess_BFGS, scaling_array_BFGS, parallel=proc_parallel, n_cores=cores,
                                              method='BFGS', verbose=verbose, plot=False) #fit the simulations
    fit_mean, fit_var, fit_std = mc_obj.compute_statistics(theta_fit) #compute the fit statistics
    relat_bias = 100*np.abs(fit_mean-(theta*scaling_array_BFGS))/(theta*scaling_array_BFGS)
    params = fitting_utils.function_handlers[model.lower()]["fit_params"]

    print(f'[IdentifiabilityAnalysis] Uncertainty:')
    print("-" * 72)
    print(f"{'Parameter':<12} | {'Theta REAL':<15} | {'Theta mean':<15} | {'Theta std.':<15} | {'Rel. bias':<12} | {'CRLB':<12}")
    print("-" * 72)
    for name, val_real, val_phys, err_abs, rel_bias, err_pct in zip(params, theta*scaling_array_BFGS,
                                                fit_mean, fit_std, relat_bias, sens_obj.rel_std):
        print(f"{name:<12} | {val_real} | {val_phys} | ± {err_abs} | ± {rel_bias} | {err_pct}")
    print("-" * 72)

    #Nyquist plots with measured and fitted data
    fig, ax = plt.subplots()
    leg = []
    ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue", s=20)
    leg.append('measured')
    ax.plot(fit_params_BFGS.opt_fit.real, -fit_params_BFGS.opt_fit.imag, color="tab:orange", linewidth=2.)
    leg.append(f'L-BFGS: χ² = {fit_params_BFGS.chi_square}')
    plt.xlabel("ℜ(Z) [Ω]")
    plt.ylabel("-ℑ(Z) [Ω]")
    plt.legend(leg)
    plt.grid()
    plt.show()

    #plot the real part of the computed sensitivity
    plt.figure()
    plot_params = fitting_utils.function_handlers[model.lower()]["fit_params"]
    for i in range(len(plot_params)):
        plt.semilogx(freqs, np.real(S_theta[:,i]), label=plot_params[i])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Re(S^Z_theta)')
    plt.legend()
    plt.grid()
    plt.show()

    #plot the imaginary part of the computed sensitivity
    plt.figure()
    for i in range(len(plot_params)):
        plt.semilogx(freqs, np.imag(S_theta[:,i]), label=plot_params[i])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Im(S^Z_theta)')
    plt.legend()
    plt.grid()
    plt.show()
