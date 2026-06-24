import os
import sys

#ensure framework is identified when running in terminal
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from framework import fitting_utils, file_lcr, characterization_utils
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#load impedance spectroscopy data
#note: in this showcase script, the raw data is already framework-compliant due to its format!
filename = './data/showcase/E4980_LCR_data.csv' #arbitrary EIS data acquired on an E4980 LCR meter
data = file_lcr.read(filename, n_samples=3, electrode="cell", acquisition_mode="spectrum", aggregate=np.mean) #object of the SpectroscopyData class

#ECM setup
#note: in this example, the chosen circuit was extracted from:
# J. P. N. Longo et al., “Sensing Hydrates in Pipes by a Combined Electrical and Optical Fiber Sensor”.
# IEEE Sensors Journal, vol. 20, pp. 5012–5018, 2020.
# doi: 10.1109/JSEN.2020.2966086.
model = 'longo2020' #key to the dictionary containing the ECM parameters
print(f'[L-BFGS_showcase] {model} parameters: {fitting_utils.function_handlers[model]}')

#L-BFGS algorithm configurations
#note: the initial guess and the scaling array should match the number of parameters in the ECM!
guess = np.array([1, 1, 1, 1, 1, 1, 0.5, 1]) #initial guess for each parameter
scaling_array = np.array([1e3, 1e-7, 1e6, 1e-2, 1e3, 1e-1, 1, 1]) #scaling for magnitude-appropriate steps
obj = fitting_utils.EquivalentCircuit(model, data, data.freqs) #equivalent circuit object

#if your raw data is not framework compliant, load or compute both the real and imaginary parts of the impedance
#something like the following will work just fine:

# Rp = data.Rp #assume the resistance data comes from an arbitrary preloaded variable
# Cp = data.Cp #assume the capacitance data comes from an arbitrary preloaded variable
# freqs = data.freqs #assume the frequency data comes from an arbitrary preloaded variable
# omegas = 2*np.pi*freqs #Hz to rad/s
# z_real = Rp/(1+(omegas*Cp*Rp)**2) #real part of the impedance
# z_imag = (omegas*Cp*(Rp**2))/(1+(omegas*Cp*Rp)**2) #imaginary part of the impedance
# Z_meas = z_real-1j*z_imag #complex impedance
# Z_meas = Z_meas.astype('complex')
# obj = fitting_utils.EquivalentCircuit(model, [z_real, z_imag], freqs) #equivalent circuit object
#quick-tip: ensure the imaginary part is passed as -Im(Z)!

fit_params_BFGS = obj.fit_circuit(guess, scaling_array, method="BFGS", verbose=True) #L-BFGS fit

#Nyquist plots with measured and fitted data
fig, ax = plt.subplots()
leg = []
ax.scatter(obj.z_meas_real, obj.z_meas_imag, marker='o', color="tab:blue", s=20)
leg.append('measured')
ax.plot(fit_params_BFGS.opt_fit.real, -fit_params_BFGS.opt_fit.imag, color="tab:orange", linewidth=1.5)
leg.append(f'L-BFGS: χ² = {fit_params_BFGS.chi_square}')
plt.xlabel("Z'")
plt.ylabel("Z''")
plt.legend(leg)
plt.grid()
plt.show()