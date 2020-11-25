import numpy as np
import esag_functions as ef
import plotting_functions as pf
import matplotlib.pyplot as plt

#####################################
# 1. IID Example:

#n = 70
#theta_iid = np.array([[0], [0], [3], [0.2], [-0.05]])
#x_iid = np.ones((n, 1))
#y_iid = np.zeros([n, 3])

#for i in range(n):
#    mu_y = np.matmul(theta_iid[:3][:], x_iid[i])    
#    gam_y = np.matmul(theta_iid[3:][:], x_iid[i])    
#    theta_y = np.concatenate((mu_y, gam_y))    
#    y_iid[i][:] = ef.simulate(theta_y, 1)    


# Plot points:
#figout = pf.plot_points(y_iid, disp = True)

# MLE:
#pars0 = np.ones((5, 1), dtype = bool)
#B0 = np.array([[0], [0], [1], [0], [0]]) + np.random.normal(0, 0.01, size = (5,1))
#print(B0)
#res = ef.MLE(y_iid, x_iid, B0, pars0, {'fatol': 1e-3, 'disp': True})
#output estimate is 1 dim array - need to reshape
#B_hat = np.reshape(res.x, (5, 1))
#print(B_hat)

####################################
# 2. Linear-Spherical Example:
# We simulate from an ESAG and we fit with ESAG and IAG models:
n = 40
x_lin = np.linspace(1, n, num = n)
x_lin = np.reshape(x_lin, (n,1))

theta_lin = np.array([[0], [0.2], [1.2], [0.4], [0.1]])
y_lin = np.zeros([n, 3])

for i in range(n):
    mu_y = np.matmul(theta_lin[:3][:], x_lin[i])
    gam_y = np.matmul(theta_lin[:3][:], x_lin[i])
    theta_y = np.concatenate((mu_y, gam_y))
    y_lin[i][:] = ef.simulate(theta_y, 1)

# Plot points on sphere:
lin_out = pf.plot_points(y_lin, disp = True)

B0 = np.array([[0], [0.1], [3], [0], [0]])
pars_esag = np.ones((5, 1), dtype = bool) 
pars_iag = np.concatenate((np.ones([3, 1], dtype = bool), np.zeros([2, 1], dtype = bool)))

out_iag = ef.MLE(y_lin, x_lin, B0, pars_iag, opts = {'fatol' : 1e-5, 'disp' : True})
B_hat_iag_mle = np.reshape(out_iag.x, (3, 1)) #only returns the part that was updated. need to add on the fixed parts.
B_hat_iag = np.concatenate((B_hat_iag_mle, np.array([[0], [0]])))
print(B_hat_iag)

out_esag = ef.MLE(y_lin, x_lin, B_hat_iag, pars_esag, opts = {'fatol' : 1e-5, 'disp' : True})
B_hat_esag = np.reshape(out_esag.x, (5, 1))
print(B_hat_esag)

# Print likelihood from each method:
print(out_iag.fun)
print(out_esag.fun)

# Simulate values from the distributions and plot together with y_lin (add lines between points?)
# Or plot contours over the data?


#####################################
# 3. Spherical-Spherical Example:
#n = 30
#theta_x = np.array([0, 0, 1, 0.1, 0])
#x = ef.simulate(theta_x, n)

#y = np.zeros([n, 3])
#B_true = np.array([[0.5, -0.2, 0.8], [1.2, 0.5, 3.0], [1.1, -0.5, -0.2], [0.1, -0.9, 2.3], [0.65, -0.9, -2.1]])

#for i in range(n):
#    mu_y = np.matmul(B_true[:3][:], x[i])    
#    gam_y = np.matmul(B_true[3:][:], x[i])    
#    theta_y = np.concatenate((mu_y, gam_y))    
#    y[i][:] = ef.simulate(theta_y, 1)    

# Plot subset of points:
#figout = pf.plot_points(x[:10])
#figout.scatter(y[:10, 0], y[:10, 1], y[:10, 2], color="g",s=20)
#plt.legend(['', 'Covariates', 'Response'])
#plt.show()

#pars0 = np.ones((5, 3), dtype = bool)
#B0 = B_true + np.random.normal(0, 0.05, size = (5,3))
#print(B0)
#res = ef.MLE(y, x, B0, pars0)
#print(res.x)
# output estimate is 1 dim array - need to reshape
#B_hat = np.reshape(res.x, (5, 3))
#print(B_hat)
#print(res.fun)
#lik_at_opt = ef.log_likelihood(y, x, B_hat)
#print(lik_at_opt) # matches res.fun - which is expected result.

#########################################
## 4. Spherical-Spherical+Factor Covariate:





