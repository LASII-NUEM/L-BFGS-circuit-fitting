import numpy as np

def Fouquet2005(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R - CPE || (R-Zws) circuit
    '''

    # expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta)*args[1] #scaling
    R1 = theta[:,0]
    R2 = theta[:,1]
    Q = theta[:,2]
    n = theta[:,3]
    Rd = theta[:,4]
    taud = theta[:,5]

    #impedance computation
    Zws = (Rd*np.tanh((1j*omega*taud)**0.5))/((1j*omega*taud)** 0.5) #warburg short-finite impedance
    CPE = Q*((1j*omega)**n) #constant phase element
    Zb2_n = R2 + Zws #num. of the CPE || (R2 - Zws) block
    Zb2_d = 1 + R2*CPE + Zws*CPE #den. of the CPE || (R2 - Zws) block
    Zb2 = Zb2_n/Zb2_d #impedance of the CPE || (R2 - Zws) block

    return R1 + Zb2

def Longo2020(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R||C - C || (R - R||CPE) circuit
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta)*args[1] #scaling
    R1 = theta[:,0]
    tau1 = theta[:,1]
    R2 = theta[:,2]
    tau2 = theta[:,3]
    R3 = theta[:,4]
    tau3 = theta[:,5]
    n3 = theta[:,6]
    tau4 = theta[:,7]

    #impedance computation
    Z_b1 = R1/(1+1j*omega*tau1) #p(R1,C1) block
    Z_b2n = R2 + (R3/(1+(1j*omega*tau3)**n3)) #num of the p(C2, R2-p(R3, CPE)) block
    Z_b2d = 1 + 1j*omega*tau2 + (1j*omega*tau4)/(1 + (1j*omega*tau3)**n3) #den of the p(C2, R2-p(R3, CPE)) block
    Z_b2 = Z_b2n/Z_b2d #p(C2, R2-p(R3, CPE)) block

    return Z_b1 + Z_b2

def LongoAdp2020(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R||C - C || (R - R||CPE) circuit
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta)*args[1] #scaling
    R1 = theta[:,0]
    C1 = theta[:,1]
    R2 = theta[:,2]
    C2 = theta[:,3]
    R3 = theta[:,4]
    Q = theta[:,5]
    n3 = theta[:,6]

    #impedance computation
    Z_b1 = R1/(1+1j*omega*R1*C1) #p(R1,C1) block
    Z_b2n = R2 + (R3/(1+R3*Q*(1j*omega)**n3)) #num of the p(C2, R2-p(R3, CPE)) block
    Z_b2d = 1 + 1j*omega*R2*C2 + (1j*omega*C2*R3)/(1 + R3*Q*(1j*omega)**n3) #den of the p(C2, R2-p(R3, CPE)) block
    Z_b2 = Z_b2n/Z_b2d #p(C2, R2-p(R3, CPE)) block

    return Z_b1 + Z_b2

def LongoRed2020(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R||C - C || (R - R||CPE) circuit
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta)*args[1] #scaling
    C2 = theta[:,0]
    R3 = theta[:,1]
    Q = theta[:,2]
    n3 = theta[:,3]

    #impedance computation
    Z_C2 = (1j*omega)*C2
    Z_CPE = Q*(1j*omega)**n3

    return 1/((1/R3)+Z_CPE+Z_C2)

def Zurich2021(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R - CPE||Zw - R||C circuit
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta)*args[1] #scaling
    R1 = theta[:,0]
    Q = theta[:,1]
    n = theta[:,2]
    Rd = theta[:,3]
    taud = theta[:,4]
    #Zws = theta[:,3]
    R2 = theta[:,5]
    C = theta[:,6]

    #impedance computation
    Zws = (Rd*np.tanh((1j*omega*taud)**0.5))/((1j*omega*taud)** 0.5) #warburg short-finite impedance
    CPE = Q*((1j*omega)**n) #constant phase element
    Z_b2_d = 1 + CPE*Zws #den. of the CPE||Zws block
    Z_b2 = Zws/Z_b2_d #impedance of the CPE||Zws block
    Z_b3_d = 1 + 1j*omega*R2*C #den. of the R||C block
    Z_b3 = R2/Z_b3_d #impedance of the R||C block

    return R1 + Z_b2 + Z_b3

def Hong2021(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R - CPE || (R - CPE||R) circuit
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    Q1 = theta[:,1]
    n1 = theta[:,2]
    R2 = theta[:,3]
    Q2 = theta[:,4]
    n2 = theta[:,5]
    R3 = theta[:,6]

    #impedance computation
    CPE1 = Q1*((1j*omega)**n1) #constant phase element
    CPE2 = Q2*((1j * omega)**n2) #constant phase element
    Z_b2_num = R2 + (R3/(1 + R3*CPE2)) #num. of the CPE || (R - CPE||R) block
    Z_b2_den = 1 + R2*CPE1 + ((R3*CPE1)/(1 + R3*CPE2)) #den. of the CPE || (R - CPE||R) block
    Z_b2 = Z_b2_num/Z_b2_den #impedance of the CPE || (R - CPE||R) block

    return R1 + Z_b2

def Awayssa2025(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent C || (R - R||C - L) circuit
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta)*args[1] #scaling
    R1 = theta[:,0]
    R2 = theta[:,1]
    C1 = theta[:,2]
    L1 = theta[:,3]
    C2 = theta[:,4]

    #impedance computation
    tau2 = 1j*omega*R2*C1 #j2R2C1
    induct_imp = 1j*omega*L1 #impedance of the inductor
    Z_num = R1 + R2 + tau2*(R1 + induct_imp) + induct_imp #num. of the impedance equivalent circuit
    Z_den = 1 + tau2*(1 + induct_imp + (1j*omega*R1*C2)) + (1j*omega*C2)*(R1 + R2 + induct_imp) #den. of the impedance equivalent circuit

    return Z_num / Z_den

def Yang2025(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R - CPE||Zw - R||C circuit
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta)*args[1] #scaling
    R1 = theta[:,0]
    R2 = theta[:,1]
    Q = theta[:,2]
    n = theta[:,3]
    Rd = theta[:,4]
    taud = theta[:,5]
    #Zws = theta[3]

    #impedance computation
    Zws = (Rd*np.tanh((1j*omega*taud)**0.5))/((1j*omega*taud)** 0.5) #warburg short-finite impedance
    CPE = Q*((1j*omega)**n) #constant phase element
    Z_b2_d = 1 + R2*CPE #den. of the CPE||Zws block

    return R1 + R2/Z_b2_d + Zws

def Zhang2024(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R - CPE||Zw - R||C circuit
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta)*args[1] #scaling
    Q = theta[:,0]
    n = theta[:,1]
    R1 = theta[:,2]
    R2 = theta[:,3]
    Rd = theta[:,4]
    taud = theta[:,5]

    #impedance computation
    Zws = (Rd*np.tanh((1j*omega*taud)**0.5))/((1j*omega*taud)** 0.5) #warburg short-finite impedance
    CPE = Q*((1j*omega)**n) #constant phase element
    Z_b1_n = R2*(1+CPE) #num. of the first block
    Z_b1_d = 1 + CPE*(R1+R2) #den. of the first block

    return Z_b1_n/Z_b1_d + Zws

def Roberts1993(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1||C1 + R2||C2
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    C1 = theta[:,1]
    R2 = theta[:,2]
    C2 = theta[:,3]

    #impedance computation
    Z1 = R1/(1+1j*omega*R1*C1)
    Z2 = R2/(1+1j*omega*R2*C2)
    return Z1+Z2

def Constable2020(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1 + R2||CPE1
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:,np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    R2 = theta[:,1]
    tau1 = theta[:,2]
    n1 = theta[:,3]

    #impedance computation
    Z_diff1 = R2/(1 + tau1*((1j*omega)**n1))

    return R1+Z_diff1

def Constable2020_logCPE(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1 + R2||CPE1
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:,np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    R2 = theta[:,1]
    Q1 = 10**theta[:,2]
    n1 = theta[:,3]

    #impedance computation
    Z_diff1 = R2/(1 + R2*Q1*((1j*omega)**n1))

    return R1+Z_diff1

def Lu2019_Rs2CPE(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1 + R2||CPE1 + R3||CPE2
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    R2 = theta[:,1]
    tau1 = theta[:,2]
    n1 = theta[:,3]
    R3 = theta[:,4]
    tau2 = theta[:,5]
    n2 = theta[:,6]

    #impedance computation
    Z_diff1 = R2/(1 + tau1*((1j*omega)**n1))
    Z_diff2 = R3/(1 + tau2*((1j*omega)**n2))

    return R1+Z_diff1+Z_diff2

def Lu2019_2CPE(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1||CPE1 + R2||CPE2
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    tau1 = theta[:,1]
    n1 = theta[:,2]
    R2 = theta[:,3]
    tau2 = theta[:,4]
    n2 = theta[:,5]

    #impedance computation
    Z_diff1 = R1/(1 + tau1*((1j*omega)**n1))
    Z_diff2 = R2/(1 + tau2*((1j*omega)**n2))

    return Z_diff1 + Z_diff2

def Lu2019_Adp2CPE(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1||CPE1 + R2||CPE2
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    Q1 = theta[:,1]
    n1 = theta[:,2]
    R2 = theta[:,3]
    Q2 = theta[:,4]
    n2 = theta[:,5]

    #impedance computation
    Z_diff1 = R1/(1 + R1*Q1*((1j*omega)**n1))
    Z_diff2 = R2/(1 + R2*Q2*((1j*omega)**n2))

    return Z_diff1 + Z_diff2

def Lu2019_2logCPE(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1||CPE1 + R2||CPE2
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    Q1 = 10**theta[:,1]
    n1 = theta[:,2]
    R2 = theta[:,3]
    Q2 = 10**theta[:,4]
    n2 = theta[:,5]

    #impedance computation
    Z_diff1 = R1/(1 + R1*Q1*((1j*omega)**n1))
    Z_diff2 = R2/(1 + R2*Q2*((1j*omega)**n2))

    return Z_diff1 + Z_diff2

def Lu2019_CPECp(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1||CPE1 + R2||C2
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    tau1 = theta[:,1]
    n1 = theta[:,2]
    R2 = theta[:,3]
    tau2 = theta[:,4]

    #impedance computation
    Z_diff1 = R1/(1 + tau1*((1j*omega)**n1))
    Z_diff2 = R2/(1 + 1j*omega*tau2)

    return Z_diff1 + Z_diff2

def Hobbes1974(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1 || R2||C2 || C1
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    R2 = theta[:,1]
    C2 = theta[:,2]
    C1 = theta[:,3]

    #impedance computation
    inv_series_RC = (1j*omega*C2)/(1+1j*omega*R2*C2)
    Z_parallel = (1/R1)+inv_series_RC+(1j*omega*C1)

    return 1/Z_parallel

def Sun2022(theta, args):
    '''
    :param theta: list with all the candidate values
    :param args: list with all the arguments that won't be minimized
    :return: impedance for the equivalent R1 || C1
    '''

    #expand thetas into the components with scaling
    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0] #rad/s

    theta = np.array(theta) * args[1] #scaling
    R1 = theta[:,0]
    C1 = theta[:,1]

    #impedance computation

    Z_parallel = (1/R1)+(1j*omega*C1)

    return 1/Z_parallel

def Sun2022_logCPE(theta, args):
    """
    R1 + (R2 // CPE), using tau formulation:
    Z = R1 + R2 / (1 + (j*w*tau1)^n1)
    """

    if theta.ndim >= 2:
        omega = args[0][:, np.newaxis] #rad/s
    else:
        theta = np.atleast_2d(theta)
        omega = args[0]#rad/s

    theta = np.array(theta)*args[1]

    R1 = theta[:,0]
    Q1 = 10**theta[:,1]
    n1 = theta[:,2]

    return R1/(1+R1*Q1*((1j*omega)**n1))