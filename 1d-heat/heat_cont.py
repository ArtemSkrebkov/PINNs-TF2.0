import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from pyDOE import lhs
from scipy.interpolate import griddata

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
# LOCAL IMPORTS

sys.path.append("utils")
from neuralnetwork import NeuralNetwork
from logger import Logger
sys.path.insert(0, 'modules')
from steppers import euler_step
from plotting import newfig, savefig, saveResultDir


def u_0_t(t):
    return np.zeros(t.shape)


def u_1_t(t):
    return np.zeros(t.shape)


# def u_0_x(x):
#     return np.sin(np.pi * x)

def u_0_x(x):
    return 3 * np.sin(2.0 * x) - np.sin(3.0 * x)


def prep_data(N_u=None, N_f=None, x_l=0.0, x_r=1.0, t_l=0.0, t_r=1.0):
    # create grid
    t = np.linspace(t_l, t_r, 100)
    x = np.linspace(x_l, x_r, 256)

    # Meshing x and t in 2D (256,100)
    X, T = np.meshgrid(x, t)

    # Preparing the inputs x and t (meshed as X, T) for predictions
    # in one single array, as X_star
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_star = exact_solution(x, 0.0, ALPHA)
    for ti in t[1:]:
        u_star = np.hstack((u_star, exact_solution(x, ti, ALPHA)))
    u_star = u_star.flatten()[:, None]

    # Domain bounds (lowerbounds upperbounds) [x, t],
    # which are here ([-1.0, 0.0] and [1.0, 1.0])
    lb = X_star.min(axis=0)
    ub = X_star.max(axis=0)
    # Getting the initial conditions (t=0)
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = u_0_x(X[0:1, :].T)
    # Getting the lowest boundary conditions (x=0)
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = u_0_t(T[:, 0:1])
    # Getting the highest boundary conditions (x=1)
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = u_1_t(T[:, -1:])
    # Stacking them in multidimensional tensors
    # for training (X_u_train is for now the continuous boundaries)
    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])

    # Generating the x and t collocation points for f,
    # with each having a N_f size
    # We pointwise add and multiply to spread the LHS over the 2D domain
    X_f_train = lb + (ub-lb)*lhs(2, N_f)

    # Generating a uniform random sample from ints between 0,
    # and the size of x_u_train, of size N_u (initial data size)
    # and without replacement (unique)
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    # Getting the corresponding X_u_train (which is
    # now scarce boundary/initial coordinates)
    X_u_train = X_u_train[idx, :]
    # Getting the corresponding u_train
    u_train = u_train[idx, :]

    return x, t, X, T,  X_star, u_star, X_u_train, u_train, X_f_train, ub, lb


# def exact_solution(x, t, alpha):
#     f = (np.exp(-4*np.pi**2*alpha*t) * np.sin(2*np.pi*x)
#          + 2.0*(1-np.exp(-np.pi**2*alpha*t))
#          * np.sin(np.pi*x) / (np.pi**2*alpha))

#     return f

def exact_solution(x, t, alpha):
    f = 3.0 * np.exp(-4.0*alpha*t)*np.sin(2.0*x) - np.exp(-9.0*alpha*t)*np.sin(3.0*x)

    return f

class HeatTransportInformedNN(NeuralNetwork):
    def __init__(self, hp, logger, X_f, ub, lb, nu):
        super().__init__(hp, logger, ub, lb)

        self.nu = nu

        # Separating the collocation coordinates
        self.x_f = self.tensor(X_f[:, 0:1])
        self.t_f = self.tensor(X_f[:, 1:2])

    # Defining custom loss
    def loss(self, u, u_pred):
        f_pred = self.f_model()
        return tf.reduce_mean(tf.square(u - u_pred)) + \
            tf.reduce_mean(tf.square(f_pred))

    # The actual PINN
    def f_model(self):
        # Using the new GradientTape paradigm of TF2.0,
        # which keeps track of operations to get the gradient at runtime
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_f)
            tape.watch(self.t_f)
            # Packing together the inputs
            X_f = tf.stack([self.x_f[:, 0], self.t_f[:, 0]], axis=1)

            # Getting the prediction
            u = self.model(X_f)
            # Deriving INSIDE the tape (since we’ll need the x derivative of this later, u_xx)
            u_x = tape.gradient(u, self.x_f)

        # Getting the other derivatives
        u_xx = tape.gradient(u_x, self.x_f)
        u_t = tape.gradient(u, self.t_f)

        # Letting the tape go
        del tape

        nu = self.get_params(numpy=True)

        # Buidling the PINNs
        # source = 2*np.sin(np.pi*self.x_f[:, 0])
        return u_t - nu*u_xx

    def get_params(self, numpy=False):
        return self.nu

    def predict(self, X_star):
        u_star = self.model(X_star)
        f_star = self.f_model()
        return u_star.numpy(), f_star.numpy()


def rhs_centered(T, dx, alpha, source):
    nx = T.shape[0]
    f = np.empty(nx)

    f[1:-1] = alpha/dx**2 * (T[:-2] - 2*T[1:-1] + T[2:]) + source[1:-1]
    f[0] = 0.
    f[-1] = 0.

    return f


def plot_inf_cont_results(X_star, u_pred, X_u_train, u_train, Exact_u, X, T, x, t, save_path=None, save_hp=None, x_l = -1.1, x_r = 1.1, u_l = -1.1 , u_r = 1.1):
    # Interpolating the results on the whole (x,t) domain.
    # griddata(points, values, points at which to interpolate, method)
    U_pred = griddata(X_star, u_pred, (X, T), method='cubic')

    # Creating the figures
    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')

    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)

    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=1.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact_u[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([x_l, x_r])
    ax.set_ylim([u_l, u_r])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_u[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([x_l, x_r])
    ax.set_ylim([u_l, u_r])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_u[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([x_l, x_r])
    ax.set_ylim([u_l, u_r])
    ax.set_title('$t = 0.75$', fontsize = 10)

    if save_path != None and save_hp != None:
        saveResultDir(save_path, save_hp)
    else:
        plt.show()


# Physical parameters
ALPHA = 0.1                     # Heat transfer coefficient
LX = np.pi                       # Size of computational domain

# Grid parameters
NX = 50                         # number of grid points
DX = LX / (NX-1)                # grid spacing
x = np.linspace(0., LX, NX)     # coordinates of grid points

# Time parameters
T0 = 0.                         # initial time
TF = 1.                         # final time
FOURIER = 0.49                  # Fourier number
DT = FOURIER*DX**2/ALPHA        # time step
NT = int((TF-T0) / DT)          # number of time steps

# Initial condition
T0 = u_0_x(x)
source = np.zeros(x.shape)
# source = 2*np.sin(np.pi*x)      # heat source term

T_euler = np.empty((NT+1, NX))
T_euler[0] = T0.copy()


for i in range(NT):
    T_euler[i+1] = euler_step(T_euler[i], rhs_centered, DT, DX, ALPHA, source)

# HYPER PARAMETERS
hp = {}
# Data size on the solution u
hp["N_u"] = 100
# Collocation points size, where we’ll check for f = 0
hp["N_f"] = 10000
# DeepNN topology (2-sized input [x t], 8 hidden layer of 20-width,
# 1-sized output [u]
hp["layers"] = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
# hp["layers"] = [2, 100, 100, 100, 100, 1]
# Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
hp["tf_epochs"] = 100
hp["tf_lr"] = 0.03
hp["tf_b1"] = 0.9
hp["tf_eps"] = None
# Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel it)
hp["nt_epochs"] = 100
hp["nt_lr"] = 0.8
hp["nt_ncorr"] = 50
hp["log_frequency"] = 10


# generate data
# x_nn, t, X, T, X_star, u_star, X_u_train, \
#     u_train, X_f, ub, lb = prep_data(hp["N_u"], hp["N_f"])

x_nn, t, X, T, X_star, u_star, X_u_train, \
    u_train, X_f, ub, lb = prep_data(hp["N_u"], hp["N_f"], x_r=np.pi)
# create model
logger = Logger(hp)
pinn = HeatTransportInformedNN(hp, logger, X_f, ub, lb, nu=ALPHA)

# training
def error():
    u_pred, _ = pinn.predict(X_star)
    return np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)


logger.set_error_fn(error)
pinn.fit(X_u_train, u_train)

# Getting the model predictions
u_pred, _ = pinn.predict(X_star)
u_pred = u_pred.flatten()
U_pred = griddata(X_star, u_pred, (X, T), method='cubic')

# plot the solution at several times
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, T_euler[0], label='Initial condition')
ax.plot(x, T_euler[int(0.5/DT)], color='green', label='Preduction Euler')
ax.plot(x_nn, U_pred[50, :], color='brown', label='Prediction PINN')
ax.plot(x, exact_solution(x, 0.5, ALPHA), '*', label='Exact solution at $t=0.5$')


ax.set_xlabel('$x$')
ax.set_ylabel('$T$')
ax.set_title('Heat transport with forward Euler scheme'
             ' vs physical informed networks')
ax.legend()
plt.show()


Exact_u = exact_solution(X, T, ALPHA)

eqnPath = "1d-heat"
plot_inf_cont_results(X_star, u_pred.flatten(), X_u_train, u_train,
                      Exact_u, X, T, x_nn, t, save_path=eqnPath, save_hp=hp, x_l=-0.2, x_r=np.pi+0.2, u_l=-3.5, u_r=3.5)
