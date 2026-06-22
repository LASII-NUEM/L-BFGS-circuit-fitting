import numpy as np
from framework import file_lcr, file_ia, characterization_utils, fitting_utils, data_types
import pickle
import os
import time
from pathlib import Path

class MediumData:
    def __init__(self, topology:dict, data_medium:data_types.SpectroscopyData, data_air:data_types.SpectroscopyData, media:str, eps_func=characterization_utils.dielectric_params_generic, fit=True):
        '''
        :param topology: list of circuit topologies that output the modeled impedance of the expected circuit
        :param data_medium: SpectroscopyData structure for the frequency sweep in the media to be characterized
        :param data_air: SpectroscopyData structure for the frequency sweep in the air
        :param media: identifier of the media to be characterized
        :param eps_func: function to compute electrical permittivity
        '''

        #validate 'topology'
        if not isinstance(topology, dict):
            raise TypeError(f'[MediumData] "topology" must be a dictionary!')

        #validate 'data_medium'
        valid_types = [data_types.SpectroscopyData, data_types.CommercialCellData, data_types.FlangeData]
        if type(data_medium) not in valid_types:
            raise TypeError(f'[MediumData] "data_medium" must be a SpectroscopyData structure! Curr. type = {type(data_medium)}')

        #validate 'data_air'
        if not isinstance(data_air, data_types.SpectroscopyData):
            raise TypeError(f'[MediumData] "data_air" must be a SpectroscopyData structure! Curr. type = {type(data_air)}')

        self.topology = topology
        self.media = media
        self.media_obj = data_medium #forward SpectroscopyData attributes to the MediumData class
        self.eps_real, self.eps_imag = eps_func(data_medium, data_air, data_medium.freqs) #compute the permittivity based on the 'eps_func' argument
        self.z_real, self.z_imag = characterization_utils.complex_impedance(data_medium, data_medium.freqs) #compute the impedance
        self.sigma_real, self.sigma_imag = characterization_utils.complex_conductivity(data_medium, data_air, data_medium.freqs, eps_func=eps_func) #compute the conductivity

        if fit and isinstance(data_medium, data_types.SpectroscopyData):
            self.fit_data = self.batch_fit_circuit(topology, data_medium, data_medium.freqs)

    def batch_fit_circuit(self, topology:dict, data_medium:data_types.SpectroscopyData, freqs:np.ndarray):
        '''
        :param topology: list of circuit topologies that output the modeled impedance of the expected circuit
        :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
        :param freqs: array with the swept frequencies
        :return: a dictionary with the fitted data, the circuit parameters, and goodness of fit statistics
        '''

        #validate the data with the Kramers-Kronig test before fitting
        linKK_obj = fitting_utils.LinearKramersKronig(data_medium, freqs, c=0.5, max_iter=100, add_capacitor=True, verbose=False)

        if linKK_obj.chi_square > 1e-2:
            print(f'[MediumData] {self.media} failed the Linear Kramers-Kronig test: x² = {linKK_obj.chi_square}')
            fitted_circuits = {"z_meas_real": self.z_real,
                               "z_meas_imag": self.z_imag,
                               "z_hat_real_KK": linKK_obj.z_hat_real,
                               "z_hat_imag_KK": linKK_obj.z_hat_imag,
                               "chi_square_KK": linKK_obj.chi_square,
                               "params_KK": linKK_obj.fit_params,
                               "M_KK": linKK_obj.fit_components,
                               "status_KK": False}
        else:
            fitted_circuits = dict((circ_name, {"z_meas_real": self.z_real,
                                                "z_meas_imag": self.z_imag,
                                                "z_hat_real_KK": linKK_obj.z_hat_real,
                                                "z_hat_imag_KK": linKK_obj.z_hat_imag,
                                                "chi_square_KK": linKK_obj.chi_square,
                                                "params_KK": linKK_obj.fit_params,
                                                "M_KK": linKK_obj.fit_components,
                                                "status_KK": True,
                                                "z_hat_real": None,
                                                "z_hat_imag": None,
                                                "nmse": None,
                                                "params": None,
                                                "t": None})
                                                for circ_name in list(topology.keys())) #dictionary to store fitted circuits parameters and values to a dictionary
            for circuit in list(topology.keys()):
                curr_fit_attributes = topology[circuit] #extract guess and scaling from the input topology argument
                curr_fit_obj = fitting_utils.EquivalentCircuit(circuit, data_medium, freqs)

                #fit using the gradient-based method (BFGS)
                curr_fit_params_BFGS = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["ice_scale_BFGS"], method="BFGS")
                z_hat_real_BFGS = curr_fit_params_BFGS.opt_fit.real #real part of the fitting
                z_hat_imag_BFGS = curr_fit_params_BFGS.opt_fit.imag #imaginary part of the fitting
                if z_hat_imag_BFGS.ndim ==1:
                    z_hat_imag_BFGS = z_hat_imag_BFGS.reshape(len(freqs), self.media_obj.n_modes)
                    z_hat_real_BFGS = z_hat_real_BFGS.reshape(len(freqs), self.media_obj.n_modes)
                nmse_BFGS = curr_fit_params_BFGS.nmse_score
                params_BFGS = curr_fit_params_BFGS.opt_params_scaled
                t_BFGS = curr_fit_params_BFGS.t_elapsed

                #fit using the non-linear-least-squares-based method (NLLS)
                #hacky-fix: scipy's curve fit has an issue that when the total iterations are reached without convergence, it raises an error
                try:
                    curr_fit_params_NLLS = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["ice_scale_NLLS"], method="NLLS")
                    z_hat_real_NLLS = curr_fit_params_NLLS.opt_fit.real
                    z_hat_imag_NLLS = curr_fit_params_NLLS.opt_fit.imag
                    if z_hat_imag_NLLS.ndim == 1:
                        z_hat_imag_NLLS = z_hat_imag_NLLS.reshape(len(freqs), self.media_obj.n_modes)
                        z_hat_real_NLLS = z_hat_real_NLLS.reshape(len(freqs), self.media_obj.n_modes)
                    nmse_NLLS = curr_fit_params_NLLS.nmse_score
                    params_NLLS = curr_fit_params_NLLS.opt_params_scaled
                    t_NLLS = curr_fit_params_NLLS.t_elapsed
                except:
                    z_hat_real_NLLS = np.zeros(shape=(len(self.z_real), self.media_obj.n_modes))
                    z_hat_imag_NLLS = np.zeros(shape=(len(self.z_imag), self.media_obj.n_modes))
                    nmse_NLLS = 1
                    params_NLLS = np.zeros_like(params_BFGS)
                    t_NLLS = None

                # fit using the damped least-squares algorithm (DLS)
                try:
                    curr_fit_params_DLS = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["ice_scale_DLS"], method="DLS")
                    z_hat_real_DLS = curr_fit_params_DLS.opt_fit.real  # real part of the fitting
                    z_hat_imag_DLS = curr_fit_params_DLS.opt_fit.imag  # imaginary part of the fitting
                    if z_hat_imag_DLS.ndim == 1:
                        z_hat_imag_DLS = z_hat_imag_DLS.reshape(len(freqs), self.media_obj.n_modes)
                        z_hat_real_DLS = z_hat_real_DLS.reshape(len(freqs), self.media_obj.n_modes)
                    params_DLS = curr_fit_params_DLS.opt_params_scaled
                    nmse_DLS = curr_fit_params_DLS.nmse_score
                    t_DLS = curr_fit_params_DLS.t_elapsed
                except:
                    z_hat_real_DLS = np.zeros(shape=(len(self.z_real), self.media_obj.n_modes))
                    z_hat_imag_DLS = np.zeros(shape=(len(self.z_imag), self.media_obj.n_modes))
                    nmse_DLS = 1
                    params_DLS = np.zeros_like(params_BFGS)
                    t_DLS = None

                #fit using the Nelder-Mead Simplex method
                curr_fit_params_simplex = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["ice_scale_SIMPLEX"], method="Nelder-Mead")
                z_hat_real_simplex = curr_fit_params_simplex.opt_fit.real #real part of the fitting
                z_hat_imag_simplex = curr_fit_params_simplex.opt_fit.imag #imaginary part of the fitting
                if z_hat_imag_simplex.ndim == 1:
                    z_hat_imag_simplex = z_hat_imag_simplex.reshape(len(freqs), self.media_obj.n_modes)
                    z_hat_real_simplex = z_hat_real_simplex.reshape(len(freqs), self.media_obj.n_modes)
                params_simplex = curr_fit_params_simplex.opt_params_scaled
                nmse_simplex = curr_fit_params_simplex.nmse_score
                t_simplex = curr_fit_params_simplex.t_elapsed

                #store the optimization parameters into the fit dictionary
                fitted_circuits[circuit]["z_hat_real"] = np.hstack((z_hat_real_BFGS, z_hat_real_NLLS, z_hat_real_DLS, z_hat_real_simplex))
                fitted_circuits[circuit]["z_hat_imag"] = np.hstack((z_hat_imag_BFGS, z_hat_imag_NLLS, z_hat_imag_DLS, z_hat_imag_simplex))
                fitted_circuits[circuit]["nmse"] = np.vstack((nmse_BFGS, nmse_NLLS, nmse_DLS, nmse_simplex)).T
                fitted_circuits[circuit]["params"] = np.vstack((params_BFGS, params_NLLS, params_DLS, params_simplex)).T
                fitted_circuits[circuit]["t"] = np.vstack((t_BFGS, t_NLLS, t_DLS, t_simplex)).T

        return fitted_circuits

class BatchData:
    def __init__(self, base_path, topology:dict, freq_threshold=None, n_samples_freq=1, n_samples_spectrum=3, electrode="cell", hardware = "LCR", aggregate=None, eps_func=characterization_utils.dielectric_params_generic, timezone=-3, fit=True):
        '''
        :param base_path: base path where all .csv files are stored
        :param topology: list of equivalent circuits to fit the data
        :param freq_threshold: value to filter frequencies from (i.e., 100 filters from 100 Hz. OBS: None to skip filtering)
        :param n_samples_freq: samples per freerun sweep
        :param n_samples_spectrum: samples per EIS sweep
        :param electrode: which hardware Cell was used to test
        :param hardware: which hardware Equipment was used to test
        :param aggregate: how to organize the data for each mode (None as default)
        :param timezone: timezone to convert unix timestamp to human timestamp
        '''

        #validate 'base_path' argument
        if not os.path.exists(base_path):
            raise FileNotFoundError(f'[BatchData] Base path {base_path} does not exist!')

        #validate 'topology'
        if not isinstance(topology, dict):
            raise TypeError(f'[BatchData] "topology" must be a dictionary!')

        #validate 'freq_threshold'
        if not freq_threshold is None:
            if isinstance(freq_threshold, float) or isinstance(freq_threshold, int):
                if not freq_threshold >= 0:
                    raise ValueError(f'[BatchData] Frequency threshold cannot be smaller than zero!')
            else:
                raise TypeError(f'[BatchData] Frequency threshold must be an integer or float!')

        #validate electrode
        electrodetype = electrode.lower() #convert to lowercase
        valid_electrodes= ['flange', 'cell', 'custom'] #list of valid sweep types
        if electrodetype not in valid_electrodes:
            raise ValueError(f'[BatchData] electrode type = {electrodetype} not implemented! Try: {valid_electrodes}')

        hardware = hardware.lower()  # convert to lowercase
        valid_hardware = ['lcr', 'ia']  # list of valid sweep types
        if hardware not in valid_hardware:
            raise ValueError(f'[BatchData] hardware = {hardware} not implemented! Try: {valid_hardware}')

        if hardware == "lcr":
            #process reference sweeps into SpectroscopyData objects
            lcr_files = [found_file for found_file in os.listdir(base_path) if found_file.endswith('.csv')] #list all LCR-based files prior to processing
            spec_objects = {"c0": None, "c1": None, "cice": None, "cthf": None, "ctest": None} #attribute the objects to sweep-based keys
            if len(lcr_files) == 0 :
                raise ValueError(f'[BatchData] There is no (.csv) file in the {base_path} directory!')
        elif hardware == "ia":
            #process reference sweeps into SpectroscopyData objects
            ia_files = [found_file for found_file in os.listdir(base_path) if found_file.endswith('.xls')] #list all IA-based files prior to processing
            spec_objects = {"spectrum": None, "ctest": None, "cthf": None} #attribute the objects to sweep-based keys
            if len(ia_files) == 0:
                raise ValueError(f'[BatchData] There is no (.xls) file in the {base_path} directory!')
        else:
            raise ValueError(f'[BatchData] There is no file in the {base_path} directory!')

        files_to_process = lcr_files if hardware == "lcr" else ia_files

        for data_file in files_to_process:

            if "Temp" in data_file:
                continue

            filepath = os.path.join(base_path, data_file) #relative path of the EIS sweep

            try:
                eis_obj_key = Path(data_file).stem #key to extract the SpectroscopyData object and run the constructor
                eis_obj_key = eis_obj_key.replace('_', '') #remove underline to access the attributes

                if eis_obj_key == "ctest":
                    if hardware == "lcr" and electrode == "flange":
                        placeholder_temp_file = os.path.join(base_path, "c_temp.lvm")
                        spec_objects[eis_obj_key] = data_types.PHOBOSData(filepath, filename_temperature=placeholder_temp_file, n_samples=n_samples_freq, electrode=electrode, acquisition_mode="freq", aggregate=aggregate) #constructor of the PHOBOSData object

                    elif hardware == "lcr" and electrode == "cell": #implement ctest for LCR and commercial cell
                        # temperature for commercial cell is a .csv file (edit file_lvm to accept .csv temperature file)
                        spec_objects[eis_obj_key] = data_types.PHOBOSData(filepath, n_samples=n_samples_freq, electrode=electrode, acquisition_mode="freq", aggregate=aggregate) #constructor of the PHOBOSData object

                    elif hardware == "ia" and electrode == "cell": #implement ctest for IA and commercial cell
                        # temperature for commercial cell is a .csv file (edit file_lvm to accept .csv temperature file)
                        spec_objects[eis_obj_key] = data_types.IAData(filepath, n_samples=n_samples_freq, electrode=electrode, acquisition_mode="freq", aggregate=aggregate)

                    #filter frequency threshold
                    if freq_threshold:
                        freq_mask = spec_objects[eis_obj_key].freqs >= freq_threshold #frequency mask
                        spec_objects[eis_obj_key].freqs = spec_objects[eis_obj_key].freqs[freq_mask] #filter frequencies
                        spec_objects[eis_obj_key].n_freqs = len(spec_objects[eis_obj_key].freqs) #update number of swept frequencies
                        if aggregate is None:
                            spec_objects[eis_obj_key].Cp = spec_objects[eis_obj_key].Cp[:,:,:,freq_mask] #filter capacitance
                            spec_objects[eis_obj_key].Rp = spec_objects[eis_obj_key].Rp[:,:,:,freq_mask] #filter resistance
                        else:
                            spec_objects[eis_obj_key].Cp = spec_objects[eis_obj_key].Cp[:,:,freq_mask] #filter capacitance
                            spec_objects[eis_obj_key].Cp_norm = spec_objects[eis_obj_key].Cp_norm[:,:,freq_mask] #filter normalized capacitance
                            spec_objects[eis_obj_key].Cp_agg = spec_objects[eis_obj_key].Cp_agg[:,freq_mask] #filter aggregated capacitance
                            spec_objects[eis_obj_key].agg_Cp_norm = spec_objects[eis_obj_key].agg_Cp_norm[:,freq_mask] #filter aggregated normalized capacitance
                            spec_objects[eis_obj_key].Rp = spec_objects[eis_obj_key].Rp[:,:,freq_mask] #filter resistance
                            spec_objects[eis_obj_key].Rp_norm = spec_objects[eis_obj_key].Rp_norm[:,:,freq_mask] #filter normalized resistance
                            spec_objects[eis_obj_key].Rp_agg = spec_objects[eis_obj_key].Rp_agg[:,freq_mask] #filter aggregated resistance
                            spec_objects[eis_obj_key].agg_Rp_norm = spec_objects[eis_obj_key].agg_Rp_norm[:,freq_mask] #filter aggregated normalized resistance
                else:

                    if hardware == "lcr" and electrode == "flange":
                        spec_objects[eis_obj_key] = file_lcr.read(filepath, n_samples=1, electrode="flange", acquisition_mode="freq", aggregate=np.mean, timezone=timezone) # SpectroscopyData object for current EIS file

                    elif hardware == "lcr" and electrode == "cell":  # implement ctest for LCR and commercial cell
                        spec_objects[eis_obj_key] = file_lcr.read(filepath, n_samples=n_samples_spectrum,
                                                                  electrode=electrode, acquisition_mode="spectrum",
                                                                  aggregate=aggregate,
                                                                  timezone=timezone)  # SpectroscopyData object for current EIS file

                    elif hardware == "ia" and electrode == "cell":  # implement ctest for IA and commercial cell
                        ia_obj = file_ia.read(filepath)
                        spec_objects["c0"] = ia_obj["c0"]
                        spec_objects["c1"] = ia_obj["c1"]
                        spec_objects["cice"] = ia_obj["cice"]

                    #filter frequency threshold
                    if freq_threshold:
                        freq_mask = spec_objects[eis_obj_key].freqs >= freq_threshold #frequency mask
                        spec_objects[eis_obj_key].freqs = spec_objects[eis_obj_key].freqs[freq_mask] #filter frequencies
                        spec_objects[eis_obj_key].n_freqs = len(spec_objects[eis_obj_key].freqs) #update number of swept frequencies
                        spec_objects[eis_obj_key].Cp = spec_objects[eis_obj_key].Cp[freq_mask] #filter capacitance
                        spec_objects[eis_obj_key].Rp = spec_objects[eis_obj_key].Rp[freq_mask] #filter resistance

            except Exception as e:
                print(f'[BatchData] Failed "{filepath}" with {e}! Processing data without it...')

        #attribute each processed SpectroscopyData object to their equivalent MediumData object
        if spec_objects["c0"] is not None:
            self.spec_air_obj = spec_objects["c0"] #object to store air EIS data
        if spec_objects["c1"] is not None:
            self.spec_h2o_obj = MediumData(topology, spec_objects["c1"], spec_objects["c0"], "water", eps_func=eps_func, fit=fit) #object to store water EIS data
        if spec_objects["cice"] is not None:
            self.spec_ice_obj = MediumData(topology, spec_objects["cice"], spec_objects["c0"],"ice", eps_func=eps_func, fit=fit) #object to store ice EIS data
        if spec_objects["cthf"] is not None:
            self.spec_thf_obj = MediumData(topology, spec_objects["cthf"], spec_objects["c0"],"thf", eps_func=eps_func, fit=fit) #object to store ice EIS data
        if spec_objects["ctest"] is not None:
            self.freerun_obj = spec_objects["ctest"] #object to store freerun sweep data

class BatchOrganizer:
    def __init__(self, base_path, circuits:dict, freq_threshold=None, n_samples_freq=1, n_samples_spectrum=3, electrode="cell", hardware = "lcr", aggregate=None, eps_func=characterization_utils.dielectric_params_generic, timezone=-3, save=True):
        '''
        :param base_path: base path where all .csv files are stored
        :param circuits: list of equivalent circuits to fit the data
        :param freq_threshold: value to filter frequencies from (i.e., 100 filters from 100 Hz. OBS: None to skip filtering)
        :param n_samples_freq: samples per freerun sweep
        :param n_samples_spectrum: samples per EIS sweep
        :param electrode: which hardware cell was used on the test
        :param hardware: which hardware equipament was used on the test
        :param aggregate: how to organize the data for each mode (None as default)
        :param timezone: timezone to convert unix timestamp to human timestamp
        :param save: flag to enable pickle dumping
        '''

        hardware = hardware.lower()  # convert to lowercase
        valid_hardware = ['lcr', 'ia']  # list of valid sweep types
        if hardware not in valid_hardware:
            raise ValueError(f'[BatchOrganizer] hardware = {hardware} not implemented! Try: {valid_hardware}')

        if hardware == "lcr":
            #try:
            self.lcr_obj = BatchData(base_path, circuits, freq_threshold=freq_threshold,
                                        n_samples_freq=n_samples_freq, n_samples_spectrum=n_samples_spectrum, electrode=electrode, hardware=hardware,
                                        aggregate=aggregate, eps_func=eps_func, timezone=timezone) #process the batch for LCR data
            #    print(f'[BatchOrganizer] LCR batch processing finished!')
            #except Exception as e:
            #    print(f'[BatchOrganizer] LCR batch processing failed with {e}! Processing data without it...')

        elif hardware == "ia":
            try:
                self.ia_obj = BatchData(base_path, circuits, freq_threshold=freq_threshold,
                                        n_samples_freq=n_samples_freq, n_samples_spectrum=n_samples_spectrum, electrode=electrode, hardware=hardware,
                                        aggregate=aggregate, eps_func=eps_func, timezone=timezone) #process the batch for LCR data
                print(f'[BatchOrganizer] IA batch processing finished!')
            except Exception as e:
                print(f'[BatchOrganizer] IA batch processing failed with {e}! Processing data without it...')
        else:
            raise ValueError(f'[BatchOrganizer] There is none of the supported hardware {hardware}!')

        #if True, save the attributes as a pickle file
        if save:
            try:
                pickle_file = f"{Path(base_path).stem}.pkl"
                self.full_relative_path = os.path.join(base_path, pickle_file)
                with open(self.full_relative_path, 'wb') as handle:
                    pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'[BatchOrganizer] Batch processing output saved at: {self.full_relative_path}')
            except Exception as e:
                print(f'[BatchOrganizer] Failed to save batch processing output with {e}!')

class BatchImpedanceFit:
    def __init__(self, filename, circuits:dict, freq_threshold=None, n_samples=1, electrode="cell", aggregate=None, timezone=-3, save=True):
        '''
        :param filename: name of the .csv file with
        :param circuits: list of equivalent circuits to fit the data
        :param freq_threshold: value to filter frequencies from (i.e., 100 filters from 100 Hz. OBS: None to skip filtering)
        :param n_samples: samples per EIS sweep
        :param electrode: which hardware was used to acquire the signals
        :param aggregate: how to organize the data for each mode (None as default)
        :param timezone: timezone to convert unix timestamp to human timestamp
        :param save: flag to enable pickle dumping
        '''

        #validate 'filename'
        if not os.path.exists(filename):
            raise ValueError(f'[BatchImpedanceFit] File {filename} does not exist!')

        #validate 'topology'
        if not isinstance(circuits, dict):
            raise TypeError(f'[BatchImpedanceFit] "circuits" must be a dictionary!')

        #validate 'freq_threshold'
        if not freq_threshold is None:
            if isinstance(freq_threshold, float) or isinstance(freq_threshold, int):
                if not freq_threshold >= 0:
                    raise ValueError(f'[BatchImpedanceFit] Frequency threshold cannot be smaller than zero!')
            else:
                raise TypeError(f'[BatchImpedanceFit] Frequency threshold must be an integer or float!')

        #self.circuits = circuits
        self.circuits = circuits
        self.media_obj = file_lcr.read(filename, n_samples=n_samples, electrode=electrode, acquisition_mode="freq", aggregate=aggregate, timezone=timezone)

        #filter frequency threshold
        if freq_threshold:
            freq_mask = self.media_obj.freqs >= freq_threshold #frequency mask
            self.media_obj.freqs = self.media_obj.freqs[freq_mask] #filter frequencies
            self.media_obj.n_freqs = len(self.media_obj.freqs) #update number of swept frequencies
            self.media_obj.Cp = self.media_obj.Cp[:,:,freq_mask] #filter capacitance
            self.media_obj.Rp = self.media_obj.Rp[:,:,freq_mask] #filter resistance

        self.z_real, self.z_imag = characterization_utils.complex_impedance(self.media_obj, self.media_obj.freqs) #compute the complex impedance
        t_init = time.time()
        self.fit_data = self.batch_fit_circuit(self.circuits, self.z_real, self.z_imag, self.media_obj.freqs) #batch fitting of the circuits

        #if True, save the attributes as a pickle file
        if save:
            try:
                filename = filename.split('/') #split on directories
                pickle_file = f"batchfit_{filename[-2]}.pkl"
                base_path = '/'.join(filename[:-1])
                self.full_relative_path = os.path.join(base_path, pickle_file)
                with open(self.full_relative_path, 'wb') as handle:
                    pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'[BatchImpedanceFit] Batch processing output saved at: {self.full_relative_path}')
            except Exception as e:
                print(f'[BatchImpedanceFit] Failed to save batch processing output with {e}!')

        print(f'[BatchImpedanceFit] finished computing all models in {time.time() - t_init}s')

    def batch_fit_circuit(self, circuits: dict, z_real:np.ndarray, z_imag:np.ndarray, freqs: np.ndarray):
        '''
        :param circuits: list of circuit topologies that output the modeled impedance of the expected circuit
        :param z_real: the real part of the complex impedance from the measured data
        :param z_imag: the imaginary part of the complex impedance from the measured data
        :param freqs: array with the swept frequencies
        :return: a dictionary with the fitted data, the circuit parameters, and goodness of fit statistics
        '''
        n_samples = len(z_real) #number of sweeps
        n_freqs = len(freqs) #number of swept frequencies
        fitted_circuits = dict((circ_name, {"z_meas_real": np.zeros(shape=(n_samples, n_freqs)),
                                            "z_meas_imag": np.zeros(shape=(n_samples, n_freqs)),
                                            "z_hat_real_KK": np.zeros(shape=(n_samples, n_freqs)),
                                            "z_hat_imag_KK": np.zeros(shape=(n_samples, n_freqs)),
                                            "chi_square_KK": np.zeros(shape=(n_samples,)),
                                            "M_KK": np.zeros(shape=(n_samples,)),
                                            "status_KK": np.zeros(shape=(n_samples,), dtype=bool),
                                            "z_hat_real": np.zeros(shape=(n_samples, n_freqs, 4)),
                                            "z_hat_imag": np.zeros(shape=(n_samples, n_freqs, 4)),
                                            "nmse": np.zeros(shape=(n_samples, 4)),
                                            "chi_square": np.zeros(shape=(n_samples, 4)),
                                            "params": [],
                                            "t": np.zeros(shape=(n_samples, 4))})
                                            for circ_name in list(circuits.keys())) #dictionary to store fitted circuits parameters and values to a dictionary

        #process each sample separately
        for i in range(0, n_samples):
            print(f'{i+1}/{n_samples}')
            curr_z_real = z_real[i,0,:] #slice the real part
            curr_z_imag = z_imag[i,0,:] #slice the real part
            linKK_obj = fitting_utils.LinearKramersKronig([curr_z_real, curr_z_imag], freqs, c=0.5, max_iter=100, add_capacitor=True, verbose=False) #validate the data with the Kramers-Kronig test before fitting

            #thresholding
            if linKK_obj.chi_square > 1e-2:
                status = False
            else:
                status = True

            #process each equivalent circuit separately
            for circuit in list(circuits.keys()):
                curr_fit_attributes = circuits[circuit] #extract guess and scaling from the input topology argument
                curr_fit_obj = fitting_utils.EquivalentCircuit(circuit, [curr_z_real, curr_z_imag], freqs)

                #fit using the gradient-based method (BFGS)
                curr_fit_params_BFGS_ice = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["ice_scale_BFGS"], method="BFGS")
                nmse_BFGS_ice = curr_fit_params_BFGS_ice.nmse_score
                curr_fit_params_BFGS_water = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["water_scale_BFGS"], method="BFGS")
                nmse_BFGS_water = curr_fit_params_BFGS_water.nmse_score

                if nmse_BFGS_ice <= nmse_BFGS_water:
                    curr_fit_params_BFGS = curr_fit_params_BFGS_ice
                    nmse_BFGS = nmse_BFGS_ice
                else:
                    curr_fit_params_BFGS = curr_fit_params_BFGS_water
                    nmse_BFGS = nmse_BFGS_water

                z_hat_real_BFGS = curr_fit_params_BFGS.opt_fit.real #real part of the fitting
                z_hat_imag_BFGS = curr_fit_params_BFGS.opt_fit.imag #imaginary part of the fitting
                t_BFGS = curr_fit_params_BFGS.t_elapsed
                params_BFGS = curr_fit_params_BFGS.opt_params_scaled
                chi_sqr_BFGS = curr_fit_params_BFGS_water.chi_square

                #fit using the non-linear-least-squares-based method (NLLS)
                #hacky-fix: scipy's curve fit has an issue that when the total iterations are reached without convergence, it raises an error
                try:
                    curr_fit_params_NLLS = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["scale_NLLS"], method="NLLS")
                    z_hat_real_NLLS = curr_fit_params_NLLS.opt_fit.real
                    z_hat_imag_NLLS = curr_fit_params_NLLS.opt_fit.imag
                    nmse_NLLS = curr_fit_params_NLLS.nmse_score
                    chi_sqr_NLLS = curr_fit_params_NLLS.chi_square
                    t_NLLS = curr_fit_params_NLLS.t_elapsed
                    params_NLLS = curr_fit_params_NLLS.opt_params_scaled
                except:
                    z_hat_real_NLLS = np.zeros(shape=(1, len(freqs)))
                    z_hat_imag_NLLS = np.zeros(shape=(1, len(freqs)))
                    nmse_NLLS = 1
                    chi_sqr_NLLS = 1
                    t_NLLS = None
                    params_NLLS = np.zeros_like(params_BFGS)

                #fit using the damped least-squares algorithm (DLS)
                try:
                    curr_fit_params_DLS = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"], curr_fit_attributes["scale_DLS"], method="DLS")
                    z_hat_real_DLS = curr_fit_params_DLS.opt_fit.real  # real part of the fitting
                    z_hat_imag_DLS = curr_fit_params_DLS.opt_fit.imag  # imaginary part of the fitting
                    nmse_DLS = curr_fit_params_DLS.nmse_score
                    chi_sqr_DLS = curr_fit_params_DLS.chi_square
                    t_DLS = curr_fit_params_DLS.t_elapsed
                    params_DLS = curr_fit_params_DLS.opt_params_scaled
                except:
                    z_hat_real_DLS = np.zeros(shape=(1, len(freqs)))
                    z_hat_imag_DLS = np.zeros(shape=(1, len(freqs)))
                    nmse_DLS = 1
                    chi_sqr_DLS = 1
                    t_DLS = None
                    params_DLS = np.zeros_like(params_BFGS)

                #fit using the Nelder-Mead Simplex method
                curr_fit_params_simplex = curr_fit_obj.fit_circuit(curr_fit_attributes["guess"],  curr_fit_attributes["ice_scale_SIMPLEX"], method="Nelder-Mead")
                z_hat_real_simplex = curr_fit_params_simplex.opt_fit.real #real part of the fitting
                z_hat_imag_simplex = curr_fit_params_simplex.opt_fit.imag #imaginary part of the fitting
                nmse_simplex = curr_fit_params_simplex.nmse_score
                chi_sqr_simplex = curr_fit_params_simplex.chi_square
                t_simplex = curr_fit_params_simplex.t_elapsed
                params_simplex = curr_fit_params_simplex.opt_params_scaled

                #attribute the computed values to the circuit key at the fit circuit dictionary
                fitted_circuits[circuit]["z_meas_real"][i,:] = curr_z_real
                fitted_circuits[circuit]["z_meas_imag"][i,:] = curr_z_imag
                fitted_circuits[circuit]["z_hat_real_KK"][i,:] = linKK_obj.z_hat_real
                fitted_circuits[circuit]["z_hat_imag_KK"][i,:] = linKK_obj.z_hat_real
                fitted_circuits[circuit]["chi_square_KK"][i] = linKK_obj.chi_square
                fitted_circuits[circuit]["M_KK"][i] = linKK_obj.fit_components
                fitted_circuits[circuit]["status_KK"][i] = status
                fitted_circuits[circuit]["z_hat_real"][i,:,:] = np.vstack((z_hat_real_BFGS, z_hat_real_NLLS, z_hat_real_DLS, z_hat_real_simplex)).T
                fitted_circuits[circuit]["z_hat_imag"][i,:,:] = np.vstack((z_hat_imag_BFGS, z_hat_imag_NLLS, z_hat_imag_DLS, z_hat_imag_simplex)).T
                fitted_circuits[circuit]["nmse"][i,:] = np.vstack((nmse_BFGS, nmse_NLLS, nmse_DLS, nmse_simplex)).T
                fitted_circuits[circuit]["chi_square"][i,:] = np.vstack((chi_sqr_BFGS, chi_sqr_NLLS, chi_sqr_DLS, chi_sqr_simplex)).T
                fitted_circuits[circuit]["t"][i,:] = np.vstack((t_BFGS, t_NLLS, t_DLS, t_simplex)).T
                fitted_circuits[circuit]["params"].append([params_BFGS, params_NLLS, params_DLS, params_simplex])

        return fitted_circuits