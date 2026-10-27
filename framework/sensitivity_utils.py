import numpy as np
from framework import fitting_utils
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats

def Jacobian_impedance(f, omegas:np.ndarray, params:np.ndarray, scaling_array=None, eps=np.sqrt(np.finfo(float).eps)):
    '''
    :param f: function to compute the numerical derivatives
    :param omegas: array with the swept angular frequencies
    :param params: array with the parameters of the model do add the step
    :param scaling_array: scaling array for the parameters
    :param eps: machine epsilon to compute the numerical derivative
    :return: numerical jacobian of the impedance (central diff.)
    '''

    J = np.zeros(shape=(len(omegas), len(params)), dtype=complex) #matrix to store the jacobian
    for i in range(len(params)):
        step_theta_i = eps*params[i] #magnitude relevant step for the current partial derivative
        forward_diff = np.copy(params)
        forward_diff[i] += step_theta_i #theta_i+eps
        backward_diff = np.copy(params)
        backward_diff[i] -= step_theta_i #theta_i-eps
        J[:,i] = (f(forward_diff, args=[omegas, scaling_array])-f(backward_diff, args=[omegas, scaling_array]))/(2*step_theta_i) #central diff.

    return J

class LocalIdentifiabilityAnalysis:
    def __init__(self, sigma_sqr:float, freqs:np.ndarray, params:np.ndarray, topology:str, eps=np.sqrt(np.finfo(float).eps), stack=True, fit_freqs=None, scaling_array=None, verbose=False):
        #initialization variables
        self.sigma_sqr = sigma_sqr #estiamted noise variance
        self.freqs = freqs #frequencies to analyze the sensitivity over
        self.params = params #parameters to compute the impedance
        self.topology = topology #which model will be used to analyze it sensivity
        self.eps = eps #machine epsilon to compute the numerical derivative
        self.fit_freqs = fit_freqs #the frequency interval that the model will be fitted (can be different from freq)
        self.scaling_array = scaling_array #scaling array for the parameters

        #handle arguments
        if self.fit_freqs is None:
            self.fit_freqs = np.copy(self.freqs)
        if self.scaling_array is None:
            self.scaling_array = np.ones_like(self.params)

        #computations
        omegas = 2*np.pi*self.freqs #angular frequency (rad/s)
        self.S_theta = self.compute_local_sensitivity(omegas, self.params, self.topology, self.scaling_array, eps=eps) #compute the local sensivity for each parameter
        self.FIM = self.compute_FisherIM(self.sigma_sqr, omegas, self.params, self.topology, self.scaling_array, eps=eps, stack=stack) #compute the Fisher Information Matrix
        self.covariance = np.linalg.inv(self.FIM) #compute the covariance matrix
        self.rel_std = np.sqrt(np.diag(self.covariance)) #relative uncertainty on the estimated parameters
        self.correlation = self.covariance/np.outer(self.rel_std, self.rel_std) #compute the correlation matrix
        self.rel_std *= scaling_array

        if verbose:
            param_names = fitting_utils.function_handlers[self.topology]["fit_params"]
            print(f'[IdentifiabilityAnalysis] Computing sensitivity with θ = {self.params}')
            print(f'[IdentifiabilityAnalysis] Condition number = {np.linalg.cond(self.FIM)}')
            print(f'[IdentifiabilityAnalysis] Parameter correlation:')
            for i in range(len(params)):
                for j in range(i + 1, len(params)):
                    if abs(self.correlation[i,j])>0.95:
                        print(f"High correlation between {param_names[i]} and {param_names[j]}: {self.correlation[i,j]}")

    def compute_local_sensitivity(self, omegas:np.ndarray, params:np.ndarray, topology:str, scaling_array:np.ndarray, eps=np.sqrt(np.finfo(float).eps)):
        '''
        :param omegas: angular frequencies to analyze the sensitivity over
        :param params: parameters to compute the impedance
        :param topology: which model will be used to analyze it sensitivity
        :param scaling_array: scaling array for the parameters
        :param plot: flag to plot the sensivity over frequency (or not)
        :return: the array with the computed sensitivity for each parameter over frequency
        '''

        # Based on:
        # F. Santoni et al., “A guide to equivalent circuit fitting for impedance analysis and battery state estimation”.
        # Journal of Energy Storage, vol. 82, pp. 110389, 2024.
        # doi: 10.1016/j.est.2023.110389.

        circuit_impedance = fitting_utils.function_handlers[topology]["function_ptr"] #pointer to the impedance function
        Z = circuit_impedance(params, args=[omegas, scaling_array]) #compute the impedance at the expected values
        J = Jacobian_impedance(circuit_impedance, omegas, params, scaling_array, eps=eps) #compute the numerical Jacobian
        weights = params[np.newaxis,:]/Z[:, np.newaxis] #normalization factor (theta_i/Z)
        S_theta = weights*J  #normalized sensitivity
        S_theta = S_theta.astype(complex)

        return S_theta

    def compute_FisherIM(self, sigma_sqr:float, omegas:np.ndarray, params:np.ndarray, topology:str, scaling_array:np.ndarray, eps=np.sqrt(np.finfo(float).eps), stack=True):
        '''
        :param sigma_sqr: estimated noise variance
        :param omegas: angular frequencies to analyze the sensitivity over
        :param params: parameters to compute the impedance
        :param topology: which model will be used to analyze it sensitivity
        :param scaling_array: scaling array for the parameters
        :param stack: flag to stack the real and imaginary parts (or just compute with the real part)
        :return: the Fisher Information Matrix
        '''

        circuit_impedance = fitting_utils.function_handlers[topology]["function_ptr"] #pointer to the impedance function
        J = Jacobian_impedance(circuit_impedance, omegas, params, scaling_array, eps=eps) #compute the numerical Jacobian

        #handle domain stacking
        if stack:
            J = np.vstack((J.real, J.imag)) #stack both real and imaginary parts
        else:
            J = J.real #compute only with real part

        return (1/sigma_sqr)*(J.T@J) #Fisher Information Matrix

class GlobalIdentifiabilityAnalysis:
    def __init__(self, freqs: np.ndarray, param_range: np.ndarray, n_samples:int, topology: str, scaling_array=None, parallel=False, n_cores=-1, skip_first=1):
        self.freqs = freqs #frequency space to analyze global sensitivity
        self.param_range = param_range #range of each parameter to generate quasi-random low-discrepancy samples
        self.n_samples = n_samples #number of quasi-random low-discrepancy samples
        self.topology = topology #which model will be used to analyze it sensivity
        self.topology_metadata = fitting_utils.function_handlers[self.topology] #dictionary containing the metadata of the chosen ECM
        self.scaling_array = scaling_array #scaling array for the parameters

        #global variance-based sensitivity analysis through Sobol indices and Saltelli sampling
        self.A, self.B, self.AB = self.quasi_monte_carlo_sampling(self.topology_metadata["n_params"], self.n_samples, self.param_range, skip_first=skip_first) #compute sampling matrices
        omegas = 2*np.pi*self.freqs #Hz to rad/s
        f = self.topology_metadata["function_ptr"] #pointer to the impedance function

        if not parallel:
            self.fA = np.zeros(shape=(len(omegas), self.n_samples))
            self.fB = np.zeros(shape=(len(omegas), self.n_samples))
            self.fAB = np.zeros(shape=(self.topology_metadata["n_params"], len(omegas), self.n_samples))
            for k in range(0, self.n_samples):
                self.fA[:,k], self.fB[:,k], self.fAB[:,:,k] = self.compute_sampling_impedance(f, omegas, self.A, self.B, self.AB, k, scaling_array=scaling_array)
        else:
            f_results = Parallel(n_jobs=n_cores, batch_size="auto", backend="threading")(
                delayed(self.compute_sampling_impedance)(f, omegas, self.A, self.B, self.AB, k, scaling_array)
                for k in range(0, self.n_samples)) #parallel tasks to compute each run of the Monte Carlo simulations

            self.fA = np.array([res[0] for res in f_results], dtype="complex").T #store each parallel-computed fA
            self.fB = np.array([res[1] for res in f_results], dtype="complex").T #store each parallel-computed fB
            self.fAB = np.array([res[2] for res in f_results], dtype="complex") #store each parallel-computed fAB
            self.fAB = np.transpose(self.fAB, (1, 2, 0)) #reshape to respect the expected shape

        self.var_Y = np.var(np.hstack([self.fA, self.fB]), axis=1, ddof=1) #outer variance over all possible points of Xi
        self.S = np.zeros(shape=(len(omegas), self.topology_metadata["n_params"])) #first order effect index
        self.St = np.zeros(shape=(len(omegas), self.topology_metadata["n_params"])) #total effect index
        for n in range(0,self.S.shape[1]):
            #compute first order effect index

            # Based on:
            # A. Saltelli et al., "Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index"
            # Computer Physics Communications, vol. 181. pp. 259-270, 2010.
            # doi: https://doi.org/10.1016/j.cpc.2009.09.018

            f1_diff = self.fB*(self.fAB[n]-self.fA)
            V_xi = np.mean(f1_diff, axis=1) #expected variance left if all factor but Xi are fixed
            self.S[:,n] = V_xi/self.var_Y

            #compute total effect sensitivity
            # Based on:
            # M.J.W. Jansen, "Analysis of variance designs for model output"
            # Computer Physics Communications, vol. 117, pp. 35–43, 1999
            # doi: https://doi.org/10.1016/S0010-4655(98)00154-4

            f_diff = (self.fA-self.fAB[n])**2
            E_not_xi = 0.5*np.mean(f_diff, axis=1) #expected variance left if all factors but Xi are be fixed
            self.St[:,n] = E_not_xi/self.var_Y

    def quasi_monte_carlo_sampling(self, n_params:int, n_samples:int, param_range:np.ndarray, skip_first=1):
        '''
        :param n_params: order of the model
        :param n_samples: number of quasi-Monte-Carlo samples to generate
        :param param_range: range of each parameter to analyze the global sensitivity over
        :param skip_first: drop the n first Sobol points (A. B. Owen, "On dropping the first Sobol point". 2021.)
        :return: Quasi-random low-discrepancy A, B, and AB sequences
        '''

        # Based on:
        # A. Saltelli, "Making best use of model evaluations to compute sensitivity indices".
        # Computer Physics Communications, 145(2), 280-297.
        # doi: https://doi.org/10.1016/S0010-4655(02)00280-1

        base_sqn = stats.qmc.Sobol(d=2 * n_params, scramble=True) #base sequence from scrambled Sobol
        base_sqn.fast_forward(skip_first) #drop Sobol points
        qMC_samples = base_sqn.random(n_samples) #sampling of the sequence

        #sampling matrices
        A_base = qMC_samples[:, :n_params] #copy samples to matrix A
        B_base = qMC_samples[:, n_params:] #copy samples to matrix B

        #scaled to the parameter range
        A = np.copy(A_base)
        B = np.copy(B_base)
        for j in range(len(param_range)):
            A[:,j] = param_range[j][0]+A_base[:,j]*(param_range[j][1]-param_range[j][0])
            B[:,j] = param_range[j][0]+B_base[:,j]*(param_range[j][1]-param_range[j][0])

        #A_B^{i} matrix: tensor where for each i-th matrix, the i-th column of A is substituted by the i-th column of B
        AB = np.zeros(shape=(n_params, n_samples, n_params))
        for i in range(0, n_params):
            AB[i] = np.copy(A)
            AB[i,:,i] = np.copy(B[:,i])

        return A, B, AB

    def compute_sampling_impedance(self, f, omegas:np.ndarray, A:np.ndarray, B:np.ndarray, AB:np.ndarray, sample_idx:int, scaling_array=None):
        '''
        :param f: pointer to the impedance function
        :param omegas: array with the all angular frequencies to analyze the model (rad/s)
        :param A: sampling matrix A
        :param B: sampling matrix B
        :param AB: sampling matrix AB (all columns are from A, except the i-th column that is from B)
        :param sample_idx: index of the qmc sample (Z from PHOBOS only accepts one set of parameters per computation)
        :param scaling_array: scaling array for the parameters
        :return: the absolute value of the impedance for each sampling matrix A, B, and AB at the sampling set "sample_idx"
        '''

        fA_complex = f(A[sample_idx, :], args=[omegas, scaling_array]) #compute the impedance for a parameter set of A
        fA_complex = fA_complex.astype("complex")
        fB_complex = f(B[sample_idx, :], args=[omegas, scaling_array]) #compute the impedance for a parameter set of B
        fB_complex = fB_complex.astype("complex")
        fAB_complex = np.zeros(shape=(A.shape[1], len(omegas)), dtype="complex")
        for j in range(A.shape[1]):
            fAB_complex[j,:] = f(AB[j,sample_idx,:], args=[omegas, scaling_array]) #compute the impedance for a parameter set of AB

        return np.abs(fA_complex), np.abs(fB_complex), np.abs(fAB_complex)

class MonteCarloFit:
    def __init__(self, topology: str, n_spectra: int, noise_var: float, theta: np.ndarray, scaling: np.ndarray, freqs: np.ndarray):
        '''
        :param topology: the circuit topoly that outputs the modeled impedance of the expected circuit
        :param n_spectra: number of simulated impedance spectra
        :param noise_var: noise variance of the gaussian noise
        :param theta: true parameters used to compute the impedance
        :param scaling: scaling of the initial guess
        :param freqs: array with the swept frequencies
        '''

        # validate 'topology'
        topology = topology.lower()  # convert to lower case
        valid_topologies = list(fitting_utils.function_handlers.keys())  # expected parameters for the available models
        if topology not in valid_topologies:
            raise ValueError(f'[MonteCarloFit] {topology} not implemented! Try: {valid_topologies}')
        self.topology = topology
        self.theta = theta
        self.scaling = scaling
        self.circuit_impedance = fitting_utils.function_handlers[topology.lower()]["function_ptr"]
        self.fit_method = None

        # validate 'n_spectra'
        if n_spectra <= 0:
            raise ValueError('[MonteCarloFit] n_spectra must be greater than 0!')
        self.n_spectra = int(n_spectra)
        self.noise_var = noise_var

        # validate freqs
        if not isinstance(freqs, np.ndarray):
            raise TypeError(f'[MonteCarloFit] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')
        self.freqs = freqs

        # simulate the circuit with the true parameters
        omega = 2 * np.pi * freqs  # Hz to rad/s
        self.Z_simu = self.circuit_impedance(self.theta,
                                             args=[omega, self.scaling])  # compute the true impedance based on theta
        self.Z_simu = self.Z_simu.astype("complex")

        #generate N impedance spectra with gaussian noise for the Monte Carlo Simulation

        # Based on:
        # Francesco Santoni, Alessio De Angelis, Antonio Moschitta, Paolo Carbone, Matteo Galeotti, Lucio Cinà, Corrado Giammanco, Aldo Di Carlo,
        # A guide to equivalent circuit fitting for impedance analysis and battery state estimation,
        # Journal of Energy Storage. 2024; 82:110389. https://doi.org/10.1016/j.est.2023.110389

        #zero-mean uncorrelated circularly-symmetric Gaussian noise
        noise_real_part = np.random.normal(0, self.noise_var / np.sqrt(2), size=(self.n_spectra, len(self.Z_simu)))
        noise_imag_part = np.random.normal(0, self.noise_var / np.sqrt(2), size=(self.n_spectra, len(self.Z_simu)))
        self.Z_noise = (self.Z_simu.real + noise_real_part) - 1j * (
                    self.Z_simu.imag + noise_imag_part)  # Gaussian-noise-corrupted samples
        self.Z_noise = self.Z_noise.astype("complex")

    def fit_spectrum(self, topology:str, Z:list, freqs:np.ndarray, initial_guess:np.ndarray, scaling_array:np.ndarray, method='BFGS', verbose=False):
        '''
        :param topology: the circuit topoly that outputs the modeled impedance of the expected circuit
        :param Z: list with both real and imaginary parts of the noise-corrupted impedance data
        :param freqs: array with the swept frequencies
        :param initial_guess: the initial guess for the fit to run the iterative algorithms
        :param scaling_array: scale all the search parameters to avoid exploding gradients
        :param method: which optimization algorithm will be used to fit the circuit data
        :param verbose: flag to print the statistics in the terminal
        :return: fitted impedance data and the optimal parameters of the model
        '''

        curr_obj = fitting_utils.EquivalentCircuit(topology, Z, freqs) #equivalent circuit object
        fit_params = curr_obj.fit_circuit(initial_guess, scaling_array, method=method, verbose=verbose) #optimal parameter search

        return fit_params.opt_fit.astype("complex"), fit_params.opt_params_scaled

    def fit_simulations(self, initial_guess: np.ndarray, scaling_array: np.ndarray, method: str, verbose=False, parallel=False, n_cores=-1, plot=False):
        '''
        :param initial_guess: the initial guess for the fit to run the iterative algorithms
        :param scaling_array: scale all the search parameters to avoid exploding gradients
        :param method: which optimization algorithm will be used to fit the circuit data
        :param verbose: flag to print the statistics in the terminal
        :param parallel: flag to compute the simulations in parallel in the CPU
        :param n_cores: number of cores to parallelize the simulations
        :param plot: flag to plot the simulated fittings over the simulated data
        :return the fitted impedance and the optimal parameters for each simulated spectrum
        '''

        #validate "method"
        valid_methods = ['BFGS', 'NLLS', 'DLS', 'Nelder-Mead', 'PSO']
        if method not in valid_methods:
            raise ValueError(f'[MonteCarloFit] {method} not implemented! Try: {valid_methods}')
        self.fit_method = method #update fit method

        #fit each corrupted spectra separately
        if not parallel:
            theta_fit = np.zeros(shape=(self.n_spectra, len(initial_guess))) #matrix to store the optimal values computed per iteration
            Z_fit = np.zeros(shape=np.shape(self.Z_noise), dtype="complex") #matrix to store the fittings per iteration
            for i in range(0, self.n_spectra):
                if verbose:
                    print(f'[MonteCarloFit] Spectra {i + 1}/{self.n_spectra}')

                Z_fit[i,:], theta_fit[i,:] = self.fit_spectrum(self.topology, [self.Z_noise.real[i, :], self.Z_noise.imag[i, :]], self.freqs,
                                                               initial_guess, scaling_array, method=method, verbose=verbose)
        else:
            if verbose:
                print(f'[MonteCarloFit] CPU parallel simulation with {n_cores} cores')
            fit_results = Parallel(n_jobs=n_cores, batch_size="auto", backend="threading")(
                          delayed(self.fit_spectrum)(self.topology, [self.Z_noise.real[i, :], self.Z_noise.imag[i, :]], self.freqs,
                                  initial_guess, scaling_array, self.fit_method, verbose)
                          for i in range(0, self.n_spectra)) #parallel tasks to compute each run of the Monte Carlo simulations

            Z_fit = np.array([res[0] for res in fit_results], dtype="complex") #store each parallel-computed fit
            theta_fit = np.array([res[1] for res in fit_results]) #store each parallel-computed theta

        if plot:
            plt.figure()
            leg = []
            for i in range(0, self.n_spectra):
                plt.plot(Z_fit.real[i, :], -Z_fit.imag[i, :], color='red', linewidth=1, zorder=2)
                plt.scatter(self.Z_noise.real[i, :], self.Z_noise.imag[i, :], c='black', s=8, zorder=1)
                if i == self.n_spectra - 1:
                    leg.append('Fit')
                    leg.append('Simulated Spectra')
            plt.xlabel("Z'")
            plt.ylabel("Z''")
            plt.legend(leg)
            plt.grid()
            plt.show()

        return Z_fit.astype("complex"), theta_fit

    def compute_statistics(self, theta_fit: np.ndarray):
        '''
        :param Z_fit: the fitted impedance for each simulated spectrum
        :param theta_fit: the optimal parameters for each simulated spectrum
        :return:
        '''

        #compute the average and standard error for each parameter
        theta_mean = np.mean(theta_fit, axis=0) #average values for the impedance spectra
        theta_sse = np.sum((theta_fit - theta_mean) ** 2, axis=0) #sum of squared errors
        theta_var = theta_sse / (self.n_spectra * (self.n_spectra - 1)) #variance of the mean
        theta_std_err = np.std(theta_fit, axis=0, ddof=1) #standard error

        return theta_mean, theta_var, theta_std_err
