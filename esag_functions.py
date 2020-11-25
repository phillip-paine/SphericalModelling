import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import minimize

def log_density(y, theta):
    invV = d3_parameterisation(theta[0:3], theta[3:])
    c1 = np.dot(np.matmul(y, invV[2]), y.T)
    
    if c1 > 0: # prevent imaginary values
        c2 = np.dot(y, theta[0:3].T)
        c3 = np.dot(theta[0:3], theta[0:3])
        alpha = c2 / (c1 ** (1/2))
        M = (1 + alpha ** 2) * norm.cdf(alpha) + alpha * norm.pdf(alpha)
        f = - math.log(2*math.pi) - 3/2 * math.log(c1) + 1/2 * ( (alpha) ** 2 - c3) + math.log(M)
    else:
        f = -math.inf # set to - inf for optimisation algo.

    return f

    
def log_likelihood(y, x, B):
    loglik = 0.0
    for i in range(len(y)):
        mu = np.matmul(B[:3][:], x[i])
        gam = np.matmul(B[3:][:], x[i])
        mu_gam = np.concatenate((mu, gam))
        loglik += log_density(y[i], mu_gam)    
    return loglik

 
def d3_parameterisation(mu, gamma):
    mu0 = np.linalg.norm(mu[1:]) # euclidean norm from numpy 
    hld = np.array([-mu0 ** 2 , mu[0]*mu[1] , mu[0]*mu[2] ])
    mu_length = np.linalg.norm(mu)
    major_ax = np.array([[1/(mu0*mu_length)*ele for ele in hld]])
    hld = [0 , - mu[2] , mu[1] ]
    minor_ax = np.array([[1/mu0*ele for ele in hld]])    
    invV = np.identity(3) + gamma[0]*(major_ax * major_ax.T - minor_ax * minor_ax.T) + gamma[1]*(major_ax * minor_ax.T + minor_ax * major_ax.T) + (math.sqrt(1 + gamma[0] ** 2 + gamma[1] ** 2) - 1) * (major_ax * major_ax.T + minor_ax * minor_ax.T)
    return [major_ax, minor_ax, invV]

def simulate(theta, n):
    mu = theta[0:3]
    gam = theta[3:]
    axis_out = d3_parameterisation(mu, gam)
    x = np.random.multivariate_normal(mu, np.linalg.inv(axis_out[2]), n)  
    y = np.array([x[i]/np.linalg.norm(x[i]) for i in range(len(x))])   
    return y

def opt_func(Bupdate, Bfixed, Y, X, pars_to_update):    
    Bfixed[pars_to_update] = Bupdate    
    out = - log_likelihood(Y, X, Bfixed)
    return out


def MLE(y, x, B0, pars, opts = {'xatol': 1e-5, 'disp': False}, alg = 'nelder-mead'):    
    Bupdate_ini = B0[pars]    

    if alg in ['nelder-mead', 'BFGS']:
        fun = lambda a : opt_func(a, B0, y, x, pars)
        res = minimize(fun, Bupdate_ini, method = alg, options = opts)

    else:
        print('Use of one the methods: nelder-mead and BFGS')
        res = 0

    return res
    

def inverseExponentialMap(x): # map from the sphere to the tangent plane at (0,0,1)
    mu = np.array([0,0,1])
    return [math.acos(np.matmul(np.transpose(mu), x[i]))/math.sin(math.acos(np.matmul(np.transpose(mu), x[i]))) *
        (x[i]) - math.cos(math.acos(np.matmul(np.transpose(mu), x[i]))) * mu + mu for i in range(len(x))]


def exponentialMap(x): # map from the tangent plane at (0,0,1) to the sphere
    out = np.empty([len(x), 3]) # initialise output array of points on S2
    mu = np.array([0, 0 ,1]) # point on sphere to be the mean direction
    for i in range(len(x)):
        V = np.array([x[i][0], x[i][2], 1])
        angle = np.linalg.norm(V - mu)
        out[i][:] = math.cos(angle) * mu + math.sin(angle) * (V - mu)/angle 
    
    return out

## def test_rotationalsyms(x):


def rotateOnSphere(a, b): # rotate the vector at a on the sphere to b - returns the rotation matrix
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    skew_sym = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotationMat = np.identity(3) + skew_sym + skew_sym ** 2 * (1 - c)/(s ** 2)
    return rotationMat
    
