import os
import sys

#ensure framework is identified when running in terminal
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from framework import fitting_utils, file_ucsd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif'] #font configuration (same as paper)

#input arguments
sample_file = 'CH4_GwINSTEK_LCR' #CH4 methane synthesized from salt-water
data_path = '../data/methane_hydrate/'  #path where all methane-hydrate-related files are stored
model = "Constable2020" #which ECM will be used to fit the data

#load the data into a data_types.UCSData object
filename = os.path.join(data_path, f'{sample_file}_spectrum.txt') #relative path to the CH4 hydrate data
ucsd_obj = file_ucsd.read(filename)

os.path.join('../../data/methane_hydrate/', f'CH4_GwINSTEK_LCR_spectrum.txt')

#Linear Kramers-Kronig with overfitting criterion
linKK = fitting_utils.LinearKramersKronig([ucsd_obj.z_meas_real, ucsd_obj.z_meas_imag], ucsd_obj.freqs, c=0.5, max_iter=100, add_capacitor=True, verbose=False)
print(f'[Linear Kramers-Kronig] M = {linKK.fit_components}')
print(f'[Linear Kramers-Kronig] Δ real = {np.mean(linKK.fit_residues_real)}%')
print(f'[Linear Kramers-Kronig] Δ imag = {np.mean(linKK.fit_residues_imag)}%')

#initial arguments
circuits = {"Constable2020": {"guess": np.array([1, 1, 1, 0.5]),
                              "scale_BFGS": np.array([1e3, 1e2, 1e-3, 1]),
                              "scale_NLLS": np.array([1e2, 1e2, 1e-5, 1]),
                              "scale_NelderMead": np.array([1e2, 1e2, 1e-5, 1]),
                              "scale_PSO": np.array([1e2, 1e2, 1e-5, 1])}} #dictionary with initial guess and scaling arrays

#algorithm-specific arguments
init_guess_BFGS = circuits[model]["guess"]
scaling_array_BFGS = circuits[model]["scale_BFGS"]

init_guess_NLLS = circuits[model]["guess"]
scaling_array_NLLS = circuits[model]["scale_NLLS"]

init_guess_simplex = circuits[model]["guess"]
scaling_array_simplex =  circuits[model]["scale_NelderMead"]

init_guess_PSO = circuits[model]["guess"]
scaling_array_PSO = circuits[model]["scale_PSO"]

#run electric circuit fitting with all the screened algorithms
fit_obj = fitting_utils.EquivalentCircuit(model, [ucsd_obj.z_meas_real, ucsd_obj.z_meas_imag], ucsd_obj.freqs) #equivalent circuit object
fit_params_BFGS = fit_obj.fit_circuit(init_guess_BFGS, scaling_array_BFGS, method="BFGS", verbose=True) #L-BFGS fit
fit_params_NLLS = fit_obj.fit_circuit(init_guess_NLLS, scaling_array_NLLS, method="NLLS", tol=1e-8, verbose=True) #NLLS fit
fit_params_simplex = fit_obj.fit_circuit(init_guess_simplex, scaling_array_simplex, method="Nelder-Mead", tol=1e-6, verbose=True) #Nelder-Mead fit
fit_params_PSO = fit_obj.fit_circuit(np.zeros(shape=(len(circuits[model]["guess"]),)), scaling_array_PSO, method="PSO", tol=1e-6, verbose=True) #PSO fit

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
interp_NLLS = ecm_circuit(fit_params_NLLS.opt_params, [2*np.pi*inter_freqs, scaling_array_NLLS]) #interpolated NLLS output
interp_NLLS = interp_NLLS.astype('complex')
interp_simplex = ecm_circuit(fit_params_simplex.opt_params, [2*np.pi*inter_freqs, scaling_array_simplex]) #interpolated Nelder-Mead output
interp_simplex = interp_simplex.astype('complex')
interp_PSO = ecm_circuit(fit_params_PSO.opt_params, [2*np.pi*inter_freqs, scaling_array_PSO]) #interpolated PSO output
interp_PSO = interp_PSO.astype('complex')

#plot the validation result
plt.figure()
leg = []
plt.scatter(linKK.z_meas_real, linKK.z_meas_imag, marker='o', color="tab:blue")
plt.plot(linKK.z_hat_real, -linKK.z_hat_imag, color='tab:orange')
leg.append(f'M = {linKK.fit_components}')
plt.xlabel("ℜ(Z) [Ω]")
plt.ylabel("-ℑ(Z) [Ω]")
plt.legend(leg)
plt.grid()

#Nyquist plots with measured and fitted data
fig, ax = plt.subplots()
leg = []
ax.scatter(fit_obj.z_meas_real, fit_obj.z_meas_imag, marker='o', color="tab:blue", s=20)
leg.append('measured')
ax.plot(interp_BFGS.real, -interp_BFGS.imag, color="tab:orange", linewidth=2.)
leg.append(f'L-BFGS: χ² = {fit_params_BFGS.chi_square}')
ax.plot(interp_NLLS.real, -interp_NLLS.imag, color="tab:green", linewidth=.75)
leg.append(f'NLLS: χ² = {fit_params_NLLS.chi_square}')
ax.plot(interp_simplex.real, -interp_simplex.imag, color="tab:red", linewidth=.75)
leg.append(f'Nelder-Mead Simplex: χ² = {fit_params_simplex.chi_square}')
ax.plot(interp_PSO.real, -interp_PSO.imag, color="tab:brown", linewidth=.75)
leg.append(f'PSO: χ² = {fit_params_PSO.chi_square}')
plt.xlabel("ℜ(Z) [Ω]")
plt.ylabel("-ℑ(Z) [Ω]")
plt.legend(leg)
plt.grid()
plt.show()

#cost function over the iterations
plt.figure()
plt.semilogy(np.arange(1,len(fit_obj.costs)+1,1), fit_obj.costs)
plt.xlabel('Iteration')
plt.ylabel('SSE')
plt.grid()
plt.tight_layout()
plt.show()

#reduced size
x_size, y_size = plt.gcf().get_size_inches()
size_ratio = .5
plt.figure(figsize=(x_size*size_ratio, y_size*size_ratio))
plt.semilogy(np.arange(1,len(fit_obj.costs)+1,1), fit_obj.costs)
plt.xlabel('Iteration')
plt.ylabel('SSE')
plt.grid()
plt.tight_layout()
plt.show()