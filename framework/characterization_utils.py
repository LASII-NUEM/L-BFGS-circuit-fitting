import numpy as np
from framework import data_types

def dielectric_params_generic(data_medium:data_types.SpectroscopyData, data_air:data_types.SpectroscopyData, freqs:np.ndarray):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :return: the real and complex parts of the dielectric parameters
    '''

    #validate data_medium
    expected_types = [data_types.SpectroscopyData, data_types.CommercialCellData, data_types.PHOBOSData]
    if type(data_medium) not in expected_types:
        raise TypeError(f'[dielectric_params_generic] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    if type(data_air) not in expected_types:
        raise TypeError(f'[dielectric_params_generic] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs
    if type(freqs) != np.ndarray:
        raise TypeError(f'[dielectric_params_generic] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')

    omegas = 2 * np.pi * freqs  # Hz to rad/s

    eps_line_medium = data_medium.Cp/data_air.Cp #real part of the permittivity
    if (data_medium.Rp.ndim==1)&(data_air.Cp.ndim==1):
        eps_2line_medium = 1/(omegas*data_medium.Rp*data_air.Cp) #imaginary part of the permittivity
    elif (data_medium.Rp.ndim > 1) & (data_air.Cp.ndim > 1):
        eps_2line_medium = 1/(omegas[:,np.newaxis]*data_medium.Rp*data_air.Cp) #imaginary part of the permittivity

    return eps_line_medium, eps_2line_medium

def dielectric_params_corrected(data_medium:data_types.SpectroscopyData, data_air:data_types.SpectroscopyData, freqs:np.ndarray):
    '''
    : Function specifically for commercial cell
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :return: the real and complex parts of the dielectric parameters, corrected for the commercial cell
    '''

    #validate data_medium
    expected_types = [data_types.SpectroscopyData, data_types.CommercialCellData]
    if type(data_medium) not in expected_types:
        raise TypeError(f'[dielectric_params_corrected] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    expected_types = [data_types.SpectroscopyData, data_types.CommercialCellData]
    if type(data_air) not in expected_types:
        raise TypeError(
            f'[dielectric_params_corrected] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs
    if type(freqs) != np.ndarray:
        raise TypeError(f'[dielectric_params_corrected] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')

    omegas = 2*np.pi*freqs #Hz to rad/s
    eps_line_medium = data_medium.Cp/data_air.Cp #real part of the permittivity
    eps_2line_medium = 1/(omegas*data_medium.Rp*data_air.Cp) #imaginary part of the permittivity
    eps_medium = (eps_line_medium - 1j*eps_2line_medium).astype('complex') #permittivity
    alpha_cell = (100*np.abs(eps_medium))/(97.0442*np.abs(eps_medium)+2.9558) #correction factor
    eps_medium = alpha_cell*eps_medium

    return np.real(eps_medium), np.abs(np.imag(eps_medium))

def dielectric_params_Artemov2013(freqs: np.ndarray, medium='ice'):
    '''
    :param freqs: array with the swept frequencies
    :param medium: which medium to generate the reference curves
    :return: the real and complex parts of the ideal dielectric parameters for the characterized medium
    '''

    #V.G. Artemov, A.A. Volkov, 2013.
    #Water and Ice Dielectric Spectra at 0°C

    #validate freqs
    if type(freqs) != np.ndarray:
        raise TypeError(f'[dielectric_params_Artemov2013] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')

    #validate medium
    medium = medium.lower() #convert to lowercase
    valid_media = ["water", "ice"]
    if medium not in valid_media:
        raise ValueError(f'[dielectric_params_Artemov2013] "medium" = {medium} not implemented! Try: {valid_media}')

    if medium == "ice":
        eps_inf = 3 #high frequency limit
        delta_epsd = 93 #contribution of the dielectric relaxation to the static dielectric constant
        tau_d = 2.2e-5 #relaxation time
    elif medium == "water":
        eps_inf = 5 #high frequency limit
        delta_epsd = 83 #contribution of the dielectric relaxation to the static dielectric constant
        tau_d = 1.7e-11 #relaxation time

    omegas = 2*np.pi*freqs #Hz to rad/s
    eps_line_ideal = eps_inf + (delta_epsd/(1 + (omegas**2)*(tau_d**2))) #real part of the permittivity
    eps_2line_ideal = (omegas*tau_d)*(delta_epsd/(1 + (omegas**2)*(tau_d**2))) #imaginary part of the permittivity

    return eps_line_ideal, eps_2line_ideal

def complex_impedance(data_medium:data_types.SpectroscopyData, freqs:np.ndarray):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :return: the real and complex parts of the impedance
    '''

    #validate data_medium
    expected_types = [data_types.SpectroscopyData, data_types.CommercialCellData, data_types.PHOBOSData]
    if type(data_medium) not in expected_types:
        raise TypeError(f'[dielectric_params_generic] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate freqs
    if type(freqs) != np.ndarray:
        raise TypeError(f'[dielectric_params_generic] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')

    omegas = 2*np.pi*freqs #Hz to rad/s
    if (data_medium.Rp.ndim==1) | (type(data_medium)==data_types.CommercialCellData):
        z_line_medium = data_medium.Rp/(1+(omegas*data_medium.Cp*data_medium.Rp)**2) #real part of the impedance
        z_2line_medium = (omegas*data_medium.Cp*(data_medium.Rp**2))/(1+(omegas*data_medium.Cp*data_medium.Rp)**2) #imaginary part of the impedance
    elif data_medium.Rp.ndim>1:
        omegas = omegas[:,np.newaxis]
        if data_medium.electrode=='flange':
            z_line_medium = data_medium.agg_Rp/(1+(omegas*data_medium.agg_Cp*data_medium.agg_Rp)**2) #real part of the impedance
            z_2line_medium = (omegas*data_medium.agg_Cp*(data_medium.agg_Rp**2))/(1+(omegas*data_medium.agg_Cp*data_medium.agg_Rp)**2) #imaginary part of the permittivity
        else:
            z_line_medium = data_medium.Rp/(1+(omegas*data_medium.Cp*data_medium.Rp)**2) #real part of the impedance
            z_2line_medium = (omegas*data_medium.Cp*(data_medium.Rp**2))/(1+(omegas*data_medium.Cp*data_medium.Rp)**2) #imaginary part of the permittivity

    return z_line_medium, z_2line_medium

def complex_conductivity(data_medium:data_types.SpectroscopyData, data_air:data_types.SpectroscopyData, freqs:np.ndarray, eps_func: dielectric_params_generic):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :return: the real and complex parts of the dielectric parameters
    '''

    #validate data_medium
    expected_types = [data_types.SpectroscopyData, data_types.CommercialCellData]
    if type(data_medium) not in expected_types:
        raise TypeError(f'[dielectric_params_generic] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    expected_types = [data_types.SpectroscopyData, data_types.CommercialCellData]
    if type(data_air) not in expected_types:
        raise TypeError(f'[dielectric_params_generic] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs
    if type(freqs) != np.ndarray:
        raise TypeError(f'[dielectric_params_generic] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')

    omegas = 2*np.pi*freqs #Hz to rad/s
    eps0 = 8.854e-12 #permittivty at vacuum
    eps_real, eps_imag = eps_func(data_medium, data_air, freqs) #compute the complex permittivity
    if (data_medium.Rp.ndim==1) | (type(data_medium)==data_types.CommercialCellData):
        sigma_line_medium = omegas*eps0*eps_imag #real conductivity
        sigma_2line_medium = omegas*eps0*eps_real #real conductivity
    elif data_medium.Rp.ndim>1:
        sigma_line_medium = omegas[:,np.newaxis]*eps0*eps_imag #real conductivity
        sigma_2line_medium = omegas[:,np.newaxis]*eps0*eps_real #real conductivity

    return sigma_line_medium, sigma_2line_medium

def dielectric_modulus(data_medium:data_types.SpectroscopyData, data_air:data_types.SpectroscopyData, freqs:np.ndarray, eps_func: dielectric_params_generic):
    '''
    :param data_medium: SpectrumData structure for the frequency sweep in the medium to be characterized
    :param data_air: SpectrumData structure for the frequency sweep in the air
    :param freqs: array with the swept frequencies
    :return: the real and complex parts of the dielectric modulus
    '''

    #validate data_medium
    expected_types = [data_types.SpectroscopyData, data_types.CommercialCellData]
    if type(data_medium) not in expected_types:
        raise TypeError(f'[dielectric_params_generic] "data_medium" must be a SpectrumData structure! Curr. type = {type(data_medium)}')

    #validate data_air
    expected_types = [data_types.SpectroscopyData, data_types.CommercialCellData]
    if type(data_air) not in expected_types:
        raise TypeError(f'[dielectric_params_generic] "data_air" must be a SpectrumData structure! Curr. type = {type(data_air)}')

    #validate freqs
    if type(freqs) != np.ndarray:
        raise TypeError(f'[dielectric_params_generic] "freqs" must be a Numpy Array! Curr. type = {type(freqs)}')

    eps_real, eps_imag = eps_func(data_medium, data_air, freqs) #compute the complex permittivity
    eps = (eps_real + 1j*eps_imag).astype('complex')
    M = 1/eps #M = 1/ε

    return M.real, M.imag