from framework import characterization_utils, data_types, equivalent_circuits, optimization_utils, script_to_overwrite_local_impedancedotpy_files
import numpy as np
from scipy.optimize import minimize, least_squares
import time

#dictionary to handle function calls -> number of expected params and the function pointer
function_handlers = {
    "longo2020": {"n_params": 8, "function_ptr": equivalent_circuits.Longo2020, "bounds": [(0,10), (0,10), (0,10), (0,10), (0,10), (0,10), (0,1), (0,10)],
                  "fit_params": ['R1', 'tau1', 'R2', 'tau2', 'R3', 'tau3', 'n3', 'tau4']},
    "longoadp2020": {"n_params": 7, "function_ptr": equivalent_circuits.LongoAdp2020, "bounds": [(0,10), (0,10), (0,10), (0,10), (0,10), (0,10), (0,1)],
                     "fit_params": ['R1', 'C1', 'R2', 'C2', 'R3', 'Q', 'n3']},
    "longolog2020": {"n_params": 7, "function_ptr": equivalent_circuits.LongoLog2020, "bounds": [(0,10), (0,10), (0,10), (0,10), (0,10), (0,10), (0,1)],
                     "fit_params": ['R1', 'C1', 'R2', 'C2', 'R3', 'Q', 'n3']},
    "longored2020": {"n_params": 4, "function_ptr": equivalent_circuits.LongoRed2020, "bounds": [(0,10), (0,10), (0,10), (0,1)],
                         "fit_params": ['C2', 'R3', 'Q', 'n3']},
    "fouquet2005": {"n_params": 6, "function_ptr": equivalent_circuits.Fouquet2005, "bounds": [(0,10), (0,10), (0,10), (0,1), (0,10), (0,10)],
                    "fit_params": ['R1', 'R2', 'Q', 'n', 'Rd', 'taud']},
    "zurich2021": {"n_params": 7, "function_ptr": equivalent_circuits.Zurich2021, "bounds": [(0,10), (0,10), (0,1), (0,10), (0,10), (0,10), (0,10)],
                   "fit_params": ["R1", "Q", "n", "Rd", "taud", "R2", "C"]},
    "awayssa2025": {"n_params": 5, "function_ptr": equivalent_circuits.Awayssa2025, "bounds": [(0,10), (0,10), (0,10), (0,10), (0,10)],
                    "fit_params": ["R1", "R2", "C1", "L1", "C2"]},
    "hong2021": {"n_params": 7, "function_ptr": equivalent_circuits.Hong2021, "bounds": [(0,10), (0,10), (0,1), (0,10), (0,10), (0,1), (0,10)],
                 "fit_params": ["R1", "Q1", "n1", "R2", "Q2", "n2", "R3"]},
    "yang2025": {"n_params": 6, "function_ptr": equivalent_circuits.Yang2025, "bounds": [(0,10), (0,10), (0,10), (0, 1), (0,10), (0,10)],
                 "fit_params": ["R1", "R2", "Q", "n", "Rd", "taud"]},
    "zhang2024": {"n_params": 6, "function_ptr": equivalent_circuits.Zhang2024, "bounds": [(0,10), (0, 1), (0,10), (0,10), (0,10), (0,10)],
                  "fit_params": ["Q", "n", "R1", "R2", "Rd", "taud"]},
    "roberts1993": {"n_params": 4, "function_ptr": equivalent_circuits.Roberts1993, "bounds": [(0,10), (0,10), (0,10), (0,10)], "fit_params": ["R1", "C1", "R2", "C2"]},
    "lu2019_rs2cpe": {"n_params": 7, "function_ptr": equivalent_circuits.Lu2019_Rs2CPE, "bounds": [(0,10), (0,10), (0,10), (0,1), (0,10), (0,10), (0,1)],
                      "fit_params": ["R1", "R2", "tau1", "n1", "R3", "tau2", "n2"]},
    "lu2019_2cpe": {"n_params": 6, "function_ptr": equivalent_circuits.Lu2019_2CPE, "bounds": [(0,10), (0,10), (0,1), (0,10), (0,10), (0,1)],
                    "fit_params": ["R1", "tau1", "n1", "R2", "tau2", "n2"]},
    "lu2019_2logcpe": {"n_params": 6, "function_ptr": equivalent_circuits.Lu2019_2logCPE, "bounds": [(0,10), (0,10), (0,1), (0,10), (0,10), (0,1)],
                            "fit_params": ["R1", "Q1", "n1", "R2", "Q2", "n2"]},
    "lu2019_cpecp": {"n_params": 5, "function_ptr": equivalent_circuits.Lu2019_CPECp, "bounds": [(0,10), (0,10), (0,1), (0,10), (0,10)],
                          "fit_params": ["R1", "tau1", "n1", "R2", "tau2"]},
    "constable2020": {"n_params": 4, "function_ptr": equivalent_circuits.Constable2020, "bounds": [(0,10), (0,10), (0,10), (0,1)],
                            "fit_params": ["R1", "R2", "tau1", "n1"]},
    "constable2020_logcpe": {"n_params": 4, "function_ptr": equivalent_circuits.Constable2020_logCPE, "bounds": [(0,np.inf), (0,np.inf), (0,np.inf), (0,1)],
                                    "fit_params": ["R1", "R2", "Q1", "n1"]},
    "hobbes1974": {"n_params": 4, "function_ptr": equivalent_circuits.Hobbes1974, "bounds": [(0,10), (0,10), (0,10), (0,10)],
                                "fit_params": ["R1", "R2", "C2", "C1"]},
    "sun2022": {"n_params": 2, "function_ptr": equivalent_circuits.Sun2022, "bounds": [(0, 10), (0, 10)],
                   "fit_params": ["R1", "C1"]},
    "sun2022_logcpe": {"n_params": 3, "function_ptr": equivalent_circuits.Sun2022_logCPE, "bounds": [(0, np.inf), (0,np.inf), (0,1)],
                           "fit_params": ["R2","Q1", "n1"]}
}

class EquivalentCircuit:
    def __init__(self, topology: str, data_medium:data_types.SpectroscopyData, freqs:np.ndarray):
        '''
        :param topology: The circuit topoly that outputs the modeled impedance of the expected circuit
        :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
        :param freqs: array with the swept frequencies
        '''

        #validate topology
        topology = topology.lower() #convert to lower case
        valid_topologies = list(function_handlers.keys()) #expected parameters for the available models
        if topology not in valid_topologies:
            raise ValueError(f'[EquivalentCircuit] {topology} not implemented! Try: {valid_topologies}')
        self.topology = topology
        self.circuit_impedance = None
        self.fit_method = None
        self.costs = []
        self.chis = []

        #validate data_medium
        expected_types = [data_types.SpectroscopyData, list]
        if type(data_medium) not in expected_types:
            raise TypeError(f'[EquivalentCircuit] "data_medium" must be {expected_types}! Curr. type = {type(data_medium)}')
        self.data_medium = data_medium

        #validate freqs
        if not isinstance(freqs, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')
        self.freqs = freqs

        #compute the measured impedance from the SpectroscopyData objects
        if type(data_medium)==data_types.SpectroscopyData:
            z_meas_real, z_meas_imag = characterization_utils.complex_impedance(data_medium, freqs)
            self.z_meas_real = z_meas_real
            self.z_meas_imag = z_meas_imag
            self.z_meas = z_meas_real-1j*z_meas_imag #complex impedance
        elif type(data_medium)==list:
            self.z_meas_real = data_medium[0]
            self.z_meas_imag = data_medium[1]
            self.z_meas = self.z_meas_real-1j*self.z_meas_imag #complex impedance

    def CUMSE(self, theta, args):
        '''
        :param z_hat: the complex impedance computed from the fitted circuit
        :param z_meas: the complex impedance measured from the real system
        :return: the mean squared error between the measured and fitted impedance values
        '''

        z_hat = self.circuit_impedance(theta, [args[1], args[2]]) #compute the model for the arguments
        z_hat = z_hat.astype('complex')
        args[0] = args[0].astype('complex')
        if z_hat.ndim >= 2:
            z_hat = z_hat.T
            SSE = np.sum(((args[0].real-z_hat.real)**2)+((args[0].imag-z_hat.imag)**2), axis=1)
        else:
            SSE = np.sum(((args[0].real-z_hat.real)**2) + ((args[0].imag-z_hat.imag)**2))

        return SSE

    def cost_callback(self, intermediate_result):
        self.costs.append(intermediate_result.fun/np.sum(self.z_meas_real**2+self.z_meas_imag**2))
        self.chis.append(intermediate_result.x.copy())

    def fit_circuit(self, initial_guess:np.ndarray, scaling_array:np.ndarray, method='BFGS', tol=1e-6, verbose=False):
        '''
        :param initial_guess: the initial guess for the fit to run the iterative algorithms
        :param scaling_array: scale all the search parameters to avoid exploding gradients
        :param method: which optimization algorithm will be used to fit the circuit data
        :param tol: tolerance for the algorithms
        :param verbose: flag to print the statistics in the terminal
        :return: the parameters the best fit the expected equivalent circuit
        '''

        #validate "method"
        valid_methods = ['BFGS', 'NLLS', 'DLS', 'Nelder-Mead', 'PSO']
        if method not in valid_methods:
            raise ValueError(f'[EquivalentCircuit] {method} not implemented! Try: {valid_methods}')
        self.fit_method = method

        #validate "initial_guess"
        if len(initial_guess) != function_handlers[self.topology]["n_params"]:
            raise ValueError(f'[EquivalentCircuit] The number of initial guess parameters do not match with the given model! Should be {function_handlers[self.topology]["n_params"]}.')

        if not isinstance(initial_guess, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "initial_guess" must be a Numpy Array! Curr. type = {type(initial_guess)}')

        #validate "scaling_array"
        if len(scaling_array) != len(initial_guess):
            raise ValueError(f'[EquivalentCircuit] Length of the scaling array do not match the length of the initial guess! {len(scaling_array)} != {len(initial_guess)}')

        if not isinstance(scaling_array, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "scaling_array" must be a Numpy Array! Curr. type = {type(scaling_array)}')

        #run the parameter search
        omega = 2*np.pi*self.freqs #Hz to rad/s
        bounds = function_handlers[self.topology]["bounds"] #optimization boundaries
        if self.fit_method == "BFGS":
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            t_init = time.time()
            fit_obj = minimize(self.CUMSE, initial_guess, args=([self.z_meas, omega, scaling_array]), bounds=bounds, method='L-BFGS-B', callback=self.cost_callback)
            t_elapsed = time.time() - t_init
            opt_fit = self.circuit_impedance(fit_obj.x, [omega, scaling_array]) #compute the circuit for the optimal values
            opt_params_scaled = fit_obj.x*scaling_array #rescale the minimized parameters
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            nrmse = self.NRMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #nrmse score for both complex parts
            chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
            mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #mae score for both complex parts

            if verbose:
                print(f'[EquivalentCircuit] Gradient-based impedance fitting:')
                print(f't = {t_elapsed} s')
                print(f'NMSE = {nmse}')
                print(f'NRMSE = {nrmse}')
                print(f'chi-square = {chisqr}')
                print(f'MAE = {mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=fit_obj.x, opt_params_scaled=opt_params_scaled, opt_cost=fit_obj.fun,
                                    opt_fit=opt_fit, nmse_score=nmse, nrmse_score=nrmse, chi_square=chisqr, mae_score=mae,
                                    n_iter=fit_obj.nit, t_elapsed=t_elapsed, costs=self.costs) #return the optimized parameters

        elif self.fit_method == "NLLS":
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            bounds = np.array(bounds) #convert the boundaries to numpy array
            bounds = ((bounds[:, 0]), (bounds[:, 1])) #curve_fit receives bounds as tuple
            t_init = time.time()
            ls_obj = least_squares(self.CUMSE, x0=initial_guess, args=([[self.z_meas, omega, scaling_array]]), bounds=bounds, max_nfev=5000) #Trust-Region-based NLLS
            t_elapsed = time.time() - t_init
            fit_params = ls_obj.x #extract the minimized parameters
            opt_fit = function_handlers[self.topology]["function_ptr"](fit_params, [omega, scaling_array]) #compute the circuit real output for the optimal values
            opt_params_scaled = fit_params*scaling_array #rescale the minimized parameters
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            nrmse = self.NRMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #nrmse score for both complex parts
            chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
            mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex"))  # mae score for both complex parts

            if verbose:
                print(f'[EquivalentCircuit] Non-linear least squares impedance fitting:')
                print(f'NLLS fit elapsed time = {t_elapsed} s')
                print(f'NMSE = {nmse}')
                print(f'NRMSE = {nrmse}')
                print(f'chi-square = {chisqr}')
                print(f'MAE = {mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=ls_obj.x, opt_params_scaled=opt_params_scaled,
                                    opt_fit=opt_fit, nmse_score=nmse, nrmse_score=nrmse, chi_square=chisqr, mae_score=mae, t_elapsed=t_elapsed) #return the optimized parameters

        elif self.fit_method == 'DLS':
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            t_init = time.time()
            fit_obj = optimization_utils.LevenbergMarquardt(self.circuit_impedance, initial_guess, args=([self.z_meas, omega, scaling_array]), tol=tol, bounds=bounds)
            t_elapsed = time.time() - t_init
            opt_fit = self.circuit_impedance(fit_obj, [omega, scaling_array]) #compute the circuit for the optimal values
            opt_params_scaled = fit_obj*scaling_array #rescale the minimized parameters
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            nrmse = self.NRMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #nrmse score for both complex parts
            chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
            mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex"))  # mae score for both complex parts

            if verbose:
                print(f'[EquivalentCircuit] Damped Least-Squares impedance fitting:')
                print(f't = {t_elapsed} s')
                print(f'NMSE = {nmse}')
                print(f'NRMSE = {nrmse}')
                print(f'chi-square = {chisqr}')
                print(f'MAE = {mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=fit_obj, opt_params_scaled=opt_params_scaled, opt_fit=opt_fit,
                                    nmse_score=nmse, chi_square=chisqr, nrmse_score=nrmse, mae_score=mae, t_elapsed=t_elapsed) #return the optimized parameters

        elif self.fit_method == "Nelder-Mead":
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            t_init = time.time()
            fit_obj = optimization_utils.NelderMeadSimplex(self.CUMSE, initial_guess, args=([self.z_meas, omega, scaling_array]), tol=tol, bounds=bounds)
            t_elapsed = time.time() - t_init
            opt_fit = self.circuit_impedance(fit_obj, [omega, scaling_array]) #compute the circuit for the optimal values
            opt_params_scaled = fit_obj*scaling_array #rescale the minimized parameters
            nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
            nrmse = self.NRMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #nrmse score for both complex parts
            chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
            mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex"))  # mae score for both complex parts

            if verbose:
                print(f'[EquivalentCircuit] Nelder-Mead Simplex impedance fitting:')
                print(f't = {t_elapsed} s')
                print(f'NMSE = {nmse}')
                print(f'NRMSE = {nrmse}')
                print(f'chi-square = {chisqr}')
                print(f'MAE = {mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=fit_obj, opt_params_scaled=opt_params_scaled, opt_fit=opt_fit,
                                    nmse_score=nmse, chi_square=chisqr, nrmse_score=nrmse, mae_score=mae, t_elapsed=t_elapsed) #return the optimized parameters

        elif self.fit_method == "PSO":
            self.circuit_impedance = function_handlers[self.topology]["function_ptr"]
            t_init = time.time()

            #compute 20 independent runs to yield the best fit through particle swarm
            #Note: this is a strategy to bypass PSO's randomness
            # Wang S-C, Liu Y-H. Research on Two-Stage Parameter Identification for Various Lithium-Ion Battery Models Using Bio-Inspired Optimization Algorithms.
            # Applied Sciences. 2026; 16(1):202. https://doi.org/10.3390/app16010202

            n_runs = 100
            best_obj = None #variable to store the 'fit_obj' with the best accuracy
            best_hat = None #varaible to store the fitted impendace with the best accuracy
            best_nmse = 10 #variable to store the best nmse
            best_chisqr = 10 #variable to store the best chi-square
            best_mae = 10 #variable to store the best mae
            for i in range(0,n_runs):
                fit_obj = optimization_utils.ParticleSwarm(self.CUMSE, function_handlers[self.topology]["n_params"], swarm_size=50, method='lbest', args=([self.z_meas, omega, scaling_array]), tol=tol, bounds=bounds)
                opt_fit = self.circuit_impedance(fit_obj, [omega, scaling_array]) #compute the circuit for the optimal values
                nmse = self.NMSE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #NMSE score for both complex parts
                chisqr = self.chi_square(self.z_meas.astype("complex"), opt_fit.astype("complex")) #chi-square score for both complex parts
                mae = self.MAE(self.z_meas.astype("complex"), opt_fit.astype("complex")) #mae score for both complex parts

                #update the variables if a new best has been achieved
                if (nmse<best_nmse) & (chisqr<best_chisqr):
                    best_obj = fit_obj
                    best_hat = opt_fit
                    best_nmse = nmse
                    best_chisqr = chisqr
                    best_mae = mae

            t_elapsed = time.time()-t_init
            best_nrmse = self.NRMSE(self.z_meas.astype("complex"), best_hat.astype("complex")) #nrmse score for both complex parts
            opt_params_scaled = best_obj*scaling_array  # rescale the minimized parameters

            if verbose:
                print(f'[EquivalentCircuit] Particle Swarm Optimization {method} impedance fitting:')
                print(f't = {t_elapsed} s')
                print(f'NMSE = {best_nmse}')
                print(f'NRMSE = {best_nrmse}')
                print(f'chi-square = {best_chisqr}')
                print(f'MAE = {best_mae}')
                fit_params = function_handlers[self.topology]["fit_params"]
                print(f'fitted params = ')
                for i in range(len(fit_params)):
                    print(f'{fit_params[i]} = {opt_params_scaled[i]}')
                print()

            return optimization_utils.OptimizerResults(opt_params=best_obj, opt_params_scaled=opt_params_scaled, opt_fit=best_hat,
                                    nmse_score=best_nmse, chi_square=best_chisqr, nrmse_score=best_nrmse, mae_score=best_mae, t_elapsed=t_elapsed) #return the optimized parameters

        else:
            raise ValueError(f'[EquivalentCircuit] method = {method} not implemented! Try: {valid_methods}')

    def NRMSE(self, z: np.ndarray, z_hat: np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: NRMSE of the fit for both real and imaginary part
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        #mean absolute percentage error
        SSE = np.sum(((z_hat.real-z.real)**2)+((z_hat.imag-z.imag)**2)) #sum of squared errors
        RMSE = np.sqrt(SSE/len(z)) #root mean squared error
        CMEAN = np.mean(z.real+z.imag) #complex mean

        return np.abs(RMSE/CMEAN)

    def NMSE(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: NMSE of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        #normalized mean squared error
        SSE = np.sum(((z_hat.real-z.real)**2) + ((z_hat.imag-z.imag)**2)) #sum of squared errors
        SSO = np.sum((z.real**2) + (z.imag**2)) #sum of squared measurements

        return SSE/SSO

    def chi_square(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: chi square of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        W = np.abs(1/(z.T @ z))
        W = W*np.ones(len(z)) #weighting matrix
        res_real = z.real-z_hat.real #real residue
        chi_square_real = res_real.T @ (W*res_real)
        res_imag = z.imag-z_hat.imag #imaginary residue
        chi_square_imag = res_imag.T @ (W*res_imag)

        return chi_square_real + chi_square_imag

    def MAE(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: chi square of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[EquivalentCircuit] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[EquivalentCircuit] "z" and "z_hat" must match in length!')

        z = z.astype("complex")
        z_hat = z_hat.astype("complex")
        err_real = np.abs(z.real-z_hat.real)/np.abs(z.real) #absolute error of the real part
        err_imag = np.abs(z.imag-z_hat.imag)/np.abs(z.imag) #absolute error of the imaginary part

        return 100*np.mean(err_real), 100*np.mean(err_imag)

class LinearKramersKronig:
    def __init__(self, data_medium:data_types.SpectroscopyData, freqs:np.ndarray, c=0.5, max_iter=100, add_capacitor=True, verbose=False):
        '''
        :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
        :param freqs: array with the swept frequencies
        :param c: threshold for overfitting criterion (mu)
        :param max_iter: maximum number of M RC pairs
        :param add_capacitor: flag to add a capacitor in series to the R_ohm+L+R_k block
        :param verbose: flag to print the statistics in the terminal
        '''

        #validate data_medium
        expected_types = [data_types.SpectroscopyData, list]
        if type(data_medium) not in expected_types:
            raise TypeError(f'[LinearKramersKronig] "data_medium" must be {expected_types}! Curr. type = {type(data_medium)}')
        self.data_medium = data_medium

        #validate freqs
        if not isinstance(freqs, np.ndarray):
            raise TypeError(f'[LinearKramersKronig] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')
        self.freqs = freqs

        #compute the measured impedance from the SpectroscopyData objects
        if type(data_medium)==data_types.SpectroscopyData:
            z_meas_real, z_meas_imag = characterization_utils.complex_impedance(data_medium, freqs)
            self.z_meas_real = z_meas_real
            self.z_meas_imag = z_meas_imag
            self.z_meas = z_meas_real - 1j*z_meas_imag #complex impedance
            self.z_meas = self.z_meas.astype("complex")
        elif type(data_medium)==list:
            self.z_meas_real = data_medium[0]
            self.z_meas_imag = data_medium[1]
            self.z_meas = self.z_meas_real-1j*self.z_meas_imag #complex impedance
            self.z_meas = self.z_meas.astype("complex")

        #fit routine
        self.tau = None #distribution of the time constants
        t_init = time.time()
        self.fit_params, self.fit_components = self.validate_data(self.z_meas, self.freqs, c=c, max_iter=max_iter, add_capacitor=add_capacitor) #run the LKK algorithm
        self.t_elapsed = time.time() - t_init #store the computation time of the algorithm
        self.z_hat_real, self.z_hat_imag = self.generate_MRCircuit(self.freqs, self.fit_params, self.tau) #compute the complex impedance for the fitted value
        self.z_hat = self.z_hat_real + 1j*self.z_hat_imag #complex impedance
        self.z_hat = self.z_hat.astype("complex")
        self.fit_residues_real, self.fit_residues_imag = self.compute_residues(self.z_meas, self. z_hat)
        self.chi_square = self.compute_chi_square(self.z_meas, self.z_hat)

        if verbose:
            print(f'[LinerKramersKronig] Data validity test:')
            if self.chi_square > 1e-2:
                print(f'Linear Kramers-Kronig test failed: x² = {self.chi_square}')
                print()
            else:
                print(f'Linear Kramers-Kronig test passed: x² = {self.chi_square}')
                print(f'Optimal M RC components = {self.fit_components}')
                print()

    def compute_residues(self, z, z_hat):
        '''
        :param z: the real measured complex impedance values
        :param z_hat: the fitted complex impedance values
        :return: the normalized residues for both real and imag parts
        '''

        z = z.astype("complex")
        z_hat = z_hat.astype("complex")
        res_real = 100*(np.abs(z.real)-np.abs(z_hat.real))/np.abs(z)
        res_imag = 100*(np.abs(z.imag)-np.abs(z_hat.imag))/np.abs(z)

        return res_real, res_imag

    def compute_mu(self, Rk):
        '''
        :param Rk: the fitted Rk components from the linear kramers-kronig test
        :return: the overfitting criterion value
        '''

        neg_Rk = Rk[Rk<0] #negative signed ohmic components
        pos_Rk = Rk[Rk>=0] #positive signed ohmic components

        return 1 - np.sum(np.abs(neg_Rk))/np.sum(np.abs(pos_Rk))

    def validate_data(self, z_meas, freqs, c=0.5, max_iter=100, add_capacitor=True):
        '''
        :param z_meas: measured complex impedance values (real and imaginary)
        :param freqs: array with the swept frequencies
        :param c: threshold for overfitting criterion (mu)
        :param max_iter: maximum number of M RC pairs
        :param add_capacitor: flag to add a capacitor in series to the R_ohm+L+R_k block
        :return: the fitted elements of the M RC circuit
        '''

        M = 0 #variable to monitor the number of RC pairs added to the circuit
        z_meas = z_meas.astype("complex") #convert to complex object (extract real and imag value separately)
        omega = 2*np.pi*freqs #rad/s to Hz
        while True:
            M += 1 #update the M RC components number

            #distribution of time constants
            self.tau = np.zeros(shape=(M,)) #M time-constants
            tau_min = 1/np.max(omega)
            tau_max = 1/np.min(omega)
            k_idx = np.arange(2, M, 1) #indexes of k to compute the time constants
            self.tau[1:-1] = 10**(np.log10(tau_min) + ((k_idx-1)/(M-1))*np.log10(tau_max/tau_min)) #log distribution of the tau values
            self.tau[0] = tau_min
            self.tau[-1] = tau_max

            #build the A matrix (Ax=b) based on the existing components
            #by default: Ẑ = R_{ohm} + R_k + L + C
            #if add_capacitor=False -> Ẑ = R_{ohm} + R_k + L
            if add_capacitor:
                A_re = np.zeros(shape=(len(omega), M+3))
                A_imag = np.zeros(shape=(len(omega), M+3))
            else:
                A_re = np.zeros(shape=(len(omega), M+2))
                A_imag = np.zeros(shape=(len(omega), M+2))

            #add the components normalized by the absolute value
            A_re[:,0] = 1/np.abs(z_meas) #R_ohm
            A_imag[:,-2] = -1/(omega*np.abs(z_meas)) #inductance
            A_imag[:,-1] = omega/np.abs(z_meas) #capacitance

            #fill the contribution from each Rk component
            rk_den = 1/(1 + 1j*omega[:,np.newaxis]@self.tau[:,np.newaxis].T) #RC element
            A_re[:,1:len(self.tau)+1] = rk_den.real/np.abs(z_meas[:,np.newaxis])
            A_imag[:,1:len(self.tau)+1] = rk_den.imag/np.abs(z_meas[:,np.newaxis])

            #fit the parameters via pseudo-inverse: x = A⁻1@b
            pi_first_half = np.linalg.inv(A_re.T@A_re + A_imag.T@A_imag) #complex (A.T@A)⁻1
            pi_second_half = A_re.T@(z_meas.real/np.abs(z_meas)) + A_imag.T@(z_meas.imag/np.abs(z_meas)) #complex A.T@b
            params = pi_first_half@pi_second_half #(A.T@A)⁻1 @ (A.T@b)

            #compute the overfitting criterion
            if add_capacitor:
                mu = self.compute_mu(params[1:-2])
            else:
                mu = self.compute_mu(params[1:-1])

            #stop criteria
            if mu <= c:
                break

            if M == max_iter:
                break

        return params, M

    def generate_MRCircuit(self, freqs, params, tau):
        '''
        :param freqs: array with the swept frequencies
        :param params: the fitted parameters from the linear kramers-kronig algorithm
        :param tau: the distribution of time constants
        :return: the real and imaginary parts of the fitted circuit (computed with the help of impedance.py circuit solver)
        '''

        #Resistive parameters
        circuit_string = f"s([R({[params[0]]},{freqs.tolist()}),"
        for Rk, tk in zip(params[1:], tau):
            circuit_string += f"K({[Rk, tk]},{freqs.tolist()}),"

        #Inductance and Capacitance parameters
        circuit_string += f"L({[params[-1]]},{freqs.tolist()}),"
        if params.size == (tau.size + 3):
            circuit_string += f"C({[1 / params[-2]]},{freqs.tolist()}),"

        circuit_string = circuit_string.strip(',')
        circuit_string += '])'
        z_hat = eval(circuit_string, script_to_overwrite_local_impedancedotpy_files.circuit_elements) #compute the complex impedance

        return z_hat.real, z_hat.imag

    def compute_chi_square(self, z:np.ndarray, z_hat:np.ndarray):
        '''
        :param z: the observed values (real measurements)
        :param z_hat: the predicted values from the fitted circuit
        :return: chi square of the fit
        '''

        #validate 'z'
        if not isinstance(z, np.ndarray):
            raise TypeError(f'[LinearKramersKronig] "z" must be a Numpy Array! Curr. type = {type(z)}')

        #validate 'z_hat'
        if not isinstance(z_hat, np.ndarray):
            raise TypeError(f'[LinearKramersKronig] "z_hat" must be a Numpy Array! Curr. type = {type(z_hat)}')

        #validate shape
        if len(z) != len(z_hat):
            raise ValueError(f'[LinearKramersKronig] "z" and "z_hat" must match in length!')

        chi_square_real = ((z.real-z_hat.real)/np.abs(z))**2
        chi_square_imag = ((z.imag-z_hat.imag)/np.abs(z))**2
        chi_square = (chi_square_real+chi_square_imag)/(z.real**2 + z.imag**2)

        return np.sum(chi_square)

def weighted_SSE(z: np.ndarray, z_hat: np.ndarray):
    '''
    :param z: the observed values (real measurements)
    :param z_hat: the predicted values from the fitted circuit
    :return: weighted sum of squared errors
    '''

    W = np.abs(1/(z.T@z))
    W = np.abs(W)*np.ones(len(z)) #weighting matrix
    res_real = z.real-z_hat.real #real residue
    chi_square_real = res_real.T@(W*res_real)
    res_imag = z.imag-z_hat.imag #imaginary residue
    chi_square_imag = res_imag.T@(W*res_imag)

    return chi_square_real+chi_square_imag