import numpy as np
from scipy.spatial.distance import cdist

class OptimizerResults:
    def __init__(self, opt_params=None, opt_params_scaled=None, opt_cost=None, opt_fit=None, nmse_score=None, nrmse_score=None, chi_square=None, mae_score=None, n_iter=None, t_elapsed=None, costs=None):
        if opt_params is not None:
            self.opt_params = opt_params
        if opt_params_scaled is not None:
            self.opt_params_scaled = opt_params_scaled
        if opt_cost is not None:
            self.opt_cost = opt_cost
        if opt_fit is not None:
            self.opt_fit = opt_fit
        if nmse_score is not None:
            self.nmse_score = nmse_score
        if nrmse_score is not None:
            self.nrmse_score = nrmse_score
        if chi_square is not None:
            self.chi_square = chi_square
        if mae_score is not None:
            self.mae_score = mae_score
        if n_iter is not None:
            self.n_iter = n_iter
        if t_elapsed is not None:
            self.t_elapsed = t_elapsed
        if costs is not None:
            self.costs = costs

def NelderMeadSimplex(cost_fun, theta:np.ndarray, args=(), alfa=1, beta=2, gamma=0.5, step=0.05, tol=1e-8, max_iter=None, bounds=None, adaptative=False):
    '''
    :param cost_fun: pointer to the cost function of the minimization problem
    :param theta: initial point for the simplex (P[0])
    :param args: list with parameters that won't be minimized but are required to compute the cost
    :param alfa: reflection coefficient
    :param beta: expansion coefficient
    :param gamma: contraction coefficient
    :param step: step to generate the Simplex from P[0]
    :param tol: tolerance of the algorithm for stop criteria (std(y) < tol)
    :param max_iter: total allowed iterations of the algorithm
    :param bounds: constraints of the problem
    :param adaptative: flag to enable parameter adaptation to the dimension of the problem
    :return the point at the simplex's vertex that minimized the cost
    '''

    #control variables of the algorithm
    n = len(theta) #points that define the simplex
    y_idx = np.arange(0,n+1,1) #indexes to extract l, h, and s

    #handle 'max_iter'
    if max_iter is None:
        max_iter = n*200

    #handle 'bounds'
    if bounds is not None:
        if isinstance(bounds, list):
            bounds = np.array(bounds) #convert to numpy array
            lower_bounds = bounds[:,0] #lower bounds for each parameter
            upper_bounds = bounds[:,1] #uppper bounds for each parameter
        else:
            lower_bounds = bounds[:,0] #lower bounds for each parameter
            upper_bounds = bounds[:,1] #uppper bounds for each parameter

    #handle 'adaptative'
    if adaptative:
        alfa = 1
        beta = 1+2/n
        gamma = 0.75-1/(2*n)

    #define the initial simplex
    step_mtx = step*np.roll(np.eye(n+1,n),0) #matrix that defines the vertices
    simplex = step_mtx + np.tile(theta[:,np.newaxis], n+1).T #apply the step

    #nelder-mead simplex algorithm
    iter = 0 #counter to monitor iterations
    while True:
        iter += 1 #update the iteration counter

        #apply constraints if required
        if bounds is not None:
            simplex = np.clip(simplex, 0, 10)

        y = cost_fun(simplex, args) #compute the cost at each vertex of the simplex

        #stop criterion
        delta = np.std(y)
        if delta < tol:
            break

        h = np.argmax(y) #index of the maximum cost
        yh = cost_fun(simplex[h,:][:,np.newaxis].T, args)
        y_idx_nH = y_idx!=h #mask to ensure i!=h in the comparison
        l = np.argmin(y) #index of the minimum cost
        yl = cost_fun(simplex[l,:][:,np.newaxis].T, args) #update the cost at the lower bound
        y_idx_cent = (y_idx!=h)&(y_idx!=l) #mask to detect second highest cost
        P_cent = np.mean(simplex[y_idx_nH], axis=0) #compute the centroid without h
        s = np.argmax(y[y_idx_cent]) #index of the second highest cost
        y_s = cost_fun(simplex[s,:][:,np.newaxis].T, args)

        #reflection
        P_r = P_cent + alfa*(P_cent-simplex[h,:])
        y_r = cost_fun(P_r[:,np.newaxis].T, args)

        #expansion
        if y_r < yl:
            P_e = P_cent + beta*(P_r-P_cent)
            y_e = cost_fun(P_e[:,np.newaxis].T, args)
            if y_e < y_r:
                simplex[h,:] = P_e
            elif y_e >= y_r:
                simplex[h,:] = P_r

        #contraction
        elif y_r >= y_s:
            if y_r < y[h]:
                simplex[h,:] = P_r

            P_c = P_cent + gamma*(simplex[h,:]-P_cent)
            y_c = cost_fun(P_c[:,np.newaxis].T, args)

            if y_c > yh:
                simplex[y_idx_cent,:] = 0.5*(simplex[y_idx_cent,:]+simplex[l, :])
                y = cost_fun(simplex, args)

            elif y_c <= yh:
                simplex[h,:] = P_c
        else:
            simplex[h,:] = P_r

        #iteration criteria
        if iter == max_iter:
            break

    opt_vertex = np.argmin(y)
    return simplex[opt_vertex,:]

def generate_Jacobian(z_fun, theta:np.ndarray, y_hat:np.ndarray, delta_params:np.ndarray, args=()):
    '''
     :param z_fun: the function to compute the impedance of the target circuit of the fitting
    :param theta: current guess for the levenberg-marquardt algorithm
    :param y_hat: predicted samples of the current iteration
    :param delta_params: fractional increment of the derivatives
    :param args: list with parameters that won't be minimized but are required to compute the impedance
    :return: the Jacobian matrix computed by the central finite difference
    '''

    m = len(y_hat) #number of samples to be fitted
    n = len(theta) #number of candidate values
    J = np.zeros(shape=(m,n)) #Jacobian matrix

    for i in range(n):
        placeholder_theta = theta[i] #store the current theta before applying central difference
        curr_step = delta_params[i]*(1+np.abs(placeholder_theta)) #step for the central difference
        if curr_step!=0:
            theta[i] = placeholder_theta+curr_step #x[i+h]
            forward_y = z_fun(theta,args) #evaluate the cost at the i+h sample
            forward_y = forward_y.astype("complex").real
            theta[i] = placeholder_theta-curr_step #x[i-h]
            backwards_y = z_fun(theta, args) #evaluate the cost at the i-h sample
            backwards_y = backwards_y.astype("complex").real
            J[:,i] = (forward_y-backwards_y)/(2*curr_step) #(x[i+h]-x[i-h])/2h

        theta[i] = placeholder_theta #rebuild the parameters array

    return J

def LevenbergMarquardt(z_fun, theta: np.ndarray, args=(), damping=1e5, gn_rate=9, gd_rate=11, tol=1e-8, max_iter=None,
                       bounds=None):
    '''
    :param z_fun: the function to compute the impedance of the target circuit of the fitting
    :param theta: initial guess for the levenberg-marquardt algorithm
    :param args: list with parameters that won't be minimized but are required to compute the impedance
    :param damping: damping coefficient
    :param gn_rate: rate to reduce the damping coefficient once the solution improves (gauss-newton scenario)
    :param gd_rate: rate to increase the damping coefficient once the solution deteriorates (gradient descent scenario)
    :param tol: tolerance of the algorithm for stop criteria (norm(J.W) < tol)
    :param max_iter: total allowed iterations of the algorithm
    :param bounds: constraints of the problem
    :return: the candidate values for Z that minimized the weighted chi-square cost function
    '''

    # control variables of the algorithm
    n_params = len(theta)  # number of candidate values
    n_points = len(args[0])  # number of samples to be fitted

    # ensure proper shapes for the signals
    args[0] = args[0].astype("complex")
    args[0] = np.atleast_1d(args[0].real).flatten()  # measured samples
    args[1] = np.atleast_1d(args[1].real).flatten()  # angular frequencies

    # handle 'max_iter'
    if max_iter is None:
        max_iter = 100 * n_params ** 2

    # handle bounds
    if bounds is not None:
        if isinstance(bounds, list):
            bounds = np.array(bounds)  # convert to numpy array
            lower_bounds = bounds[:, 0]  # lower bounds for each parameter
            upper_bounds = bounds[:, 1]  # uppper bounds for each parameter
        else:
            lower_bounds = bounds[:, 0]  # lower bounds for each parameter
            upper_bounds = bounds[:, 1]  # uppper bounds for each parameter

    # define the defaults weights based no the measured samples
    W = np.abs(1 / (args[0].T @ args[0]))  # W = 1/sum(y[i]**2)
    W *= np.ones(n_points)  # weight matrix
    delta_params = 0.001 * np.ones(n_params)  # fractional increment of the derivatives

    # initialize the DLS matrices
    y_hat = z_fun(theta, args[1:])  # compute the impedance of the circuit at the current candidates
    y_hat = y_hat.astype("complex")
    y_hat = np.atleast_1d(y_hat.real).flatten()  # ensure sizes
    J = generate_Jacobian(z_fun, theta, y_hat, delta_params, args=args[1:])  # compute the Jacobian
    res = args[0].real - y_hat  # compute the residues (y-ŷ)
    chi_sqr = res.T @ (W * res)  # sum of the weighted squared errors
    last_chi_sqr = chi_sqr  # variable to monitor the evolution of the cost function (damping factor tolerance)]
    H = J.T @ (J * W[:, np.newaxis])  # compute the Hessian (second order term)
    b = J.T @ (W * res)  # vector of coefficients

    # levenberg-marquardt algorithm
    iter = 0  # counter to monitor iterations
    while True:
        iter += 1  # update the iteration counter
        A = H + damping * np.diag(np.diag(H))  # (JᵀWJ + λ*diag(JᵀWJ)) -> Marquardt's update relationship

        # solve for the step in the candidate values with the pseudo-inverse
        try:
            h_first_half = np.linalg.inv(A.T @ A)
            h_second_half = A.T @ b
            h = h_first_half @ h_second_half
        except:
            # ensure matrix is well-conditioned (in case of singular matrices)
            while np.linalg.cond(A) > 1e15:
                A = A + 1e-6 * np.sum(np.diag(A)) / n_params * np.eye(n_params)
                h_first_half = np.linalg.inv(A.T @ A)
            h_second_half = A.T @ b
            h = h_first_half @ h_second_half

        # evaluation step
        candidate_theta = theta + h  # update the candidate guess (theta+step)
        candidate_theta = np.clip(candidate_theta, 0, 10)  # apply constraints
        y_hat = z_fun(candidate_theta, args[1:])  # compute the impedance at the candidate guess
        y_hat = y_hat.astype("complex")
        y_hat = np.atleast_1d(y_hat.real).flatten()
        res = args[0].real - y_hat  # compute the residue at the candidate guess
        chi_sqr = res.T @ (W * res)  # compute the sum of the weighted squared errors at the candidate guess

        # update step
        if chi_sqr < last_chi_sqr:
            damping /= gn_rate  # reduce the damping coefficient to enable gauss-newton (closer to the local min.)
            theta = candidate_theta  # update the guess
            last_chi_sqr = chi_sqr  # commute to store the chi-sqaure as the last valid cost
            J = generate_Jacobian(z_fun, theta, y_hat, delta_params, args=args[1:])  # update the Jacobian
            H = J.T @ (J * W[:, np.newaxis])  # update the Hessian
            b = J.T @ (W * res)  # update the vector of coefficients

            # gradient stop criteria
            if np.linalg.norm(J * W[:, np.newaxis]) < tol:
                break

            # chi-square stop criteria
            if chi_sqr < tol:
                break

        else:
            damping *= gd_rate  # increase the damping coefficient to enable gradient descent (farther from the local min.)

        # iteration stop criteria
        if iter == max_iter:
            break

    return theta

def ring_topology(swarm_positions, swarm_costs):
    '''
    :param swarm_positions: the position of the particles in the swarm
    :param swarm_costs: cost of each particle in the swarm
    :return: the local best positions and costs of each particle in the swarm
    '''

    left_neighbors_pos = np.roll(swarm_positions, 1, axis=0)
    left_neighbors_cost = np.roll(swarm_costs, 1, axis=0)
    right_neighbors_pos = np.roll(swarm_positions, -1, axis=0)
    right_neighbors_cost = np.roll(swarm_costs, -1, axis=0)
    ring_pos = np.stack([left_neighbors_pos, swarm_positions, right_neighbors_pos]) #ring topology for positions
    ring_cost = np.stack([left_neighbors_cost, swarm_costs, right_neighbors_cost]) #ring topology for costs
    local_best = np.argmin(ring_cost, axis=0) #row wise index for the local best costs
    cols = np.arange(0,ring_cost.shape[1],1) #index of the columns
    swarm_costs[:] = ring_cost[local_best,cols] #mask the costs given the index of the local best
    swarm_positions[:,:] = ring_pos[local_best,cols,:] #mask the positions given the index of the local best

    return swarm_positions, swarm_costs

def ParticleSwarm(cost_fun, n, args=(), method='gbest', swarm_size=50, c1=2, c2=2, weight=0.8, delta=0.5, tol=1e-6, max_iter=None, bounds=None):
    '''
    :param cost_fun: pointer to the cost function of the minimization problem
    :param n: number of dimensions
    :param args: list with parameters that won't be minimized but are required to compute the cost
    :param method: algorithm that sets which particle will be used as the best (lbest by default)
    :param swarm_size: number of particles in a swarm
    :param c1: cognitive acceleration
    :param c2: social acceleration
    :param weight: inertia weight
    :param delta: velocity clamping factor
    :param tol: tolerance of the algorithm for stop criteria
    :param max_iter: total allowed iterations of the algorithm
    :param bounds: constraints of the problem
    :return the best particle that minimized the cost
    '''

    #validate method
    valid_methods = ['gbest', 'lbest']
    if method not in valid_methods:
        raise ValueError(f'[ParticleSwarm] Method {method} is not valid! Try: {valid_methods}')

    #handle iteration
    if max_iter is None:
        max_iter = 400 #best performance overall

    #randomly generate the positions and velocities of the swarm
    bounds = (0,10)
    swarm_positions = np.random.uniform(bounds[0], bounds[1], size=(swarm_size, n)) #array to store the positions
    swarm_costs = cost_fun(swarm_positions, args) #compute the cost for the current particles
    swarm_velocities = np.random.uniform(bounds[0], bounds[1], size=(swarm_size, n)) #array to store the positions
    P_best = np.copy(swarm_positions) #the best position found by each particle
    cost_P_best = np.copy(swarm_costs) #the cost at the best position found by each particle

    if method == 'lbest':
        G_best, cost_G_best = ring_topology(P_best, cost_P_best) #find the best local particle in a neighborhood of 3
    elif method == 'gbest':
        idx_best = np.argmin(cost_P_best) #find the index of the best cost up until now
        G_best = P_best[idx_best,:]*np.ones_like(swarm_positions) #find the position of the best particle in the swarm

    n_iter = 0 #variable to monitor the iterations
    while True:
        #iteration stop criteria
        n_iter += 1
        if n_iter == max_iter:
            break

        r1,r2 = np.random.uniform(0, 1, 2) #random uniform values
        swarm_velocities = weight*swarm_velocities + c1*r1*(P_best-swarm_positions) + c2*r2*(G_best-swarm_positions) #evaluate the new velocity

        #apply velocity clamping
        v_max = delta*(bounds[1]-bounds[0])
        swarm_velocities[swarm_velocities>=v_max] = v_max

        swarm_positions += swarm_velocities #update the positions given the velocity
        swarm_positions = np.clip(swarm_positions, bounds[0], bounds[1]) #respect the bounds
        swarm_costs = cost_fun(swarm_positions, args) #update the costs
        cost_mask = swarm_costs<cost_P_best #find where the new positions return the better costs
        P_best[cost_mask] = swarm_positions[cost_mask] #update the best positions
        cost_P_best[cost_mask] = swarm_costs[cost_mask] #update the best costs

        if method == 'lbest':
            G_best, cost_G_best = ring_topology(P_best, cost_P_best)

            #swarm distance stop criteria
            swarm_dist = cdist(swarm_positions, swarm_positions) #compute the Euclidean distance between each particle
            if np.linalg.norm(swarm_dist[:,0])<tol:
                break

        elif method == 'gbest':
            idx_best = np.argmin(cost_P_best) #find the new minimum
            G_best = P_best[idx_best, :]*np.ones_like(swarm_positions) #update new global best

            #swarm distance stop criteria
            swarm_dist = cdist(swarm_positions, G_best) #compute the Euclidean distance between each particle and the global best
            if np.linalg.norm(swarm_dist[:,0])<tol:
                break

    return P_best[np.argmin(cost_P_best)]
