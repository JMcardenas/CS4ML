import math
import numpy as np
import tensorflow as tf 
from numpy.polynomial.hermite import hermgauss

DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)
#------------------------------------------------------------------#
# PDE functions 
#------------------------------------------------------------------#

# Set constants
pi = tf.constant(np.pi, dtype=DTYPE)
viscosity = .01/pi

# Define initial condition
def fun_u_0(x):
 return -tf.sin(pi * x)

# Define boundary condition
def fun_u_b(t, x):
 n = x.shape[0]
 return tf.zeros((n,1), dtype=DTYPE)

# Define residual of the PDE
def fun_r(t, x, u, u_t, u_x, u_xx):
 return u_t + u * u_x - viscosity * u_xx

# Define approximate solution of burgers equation
def burgers_viscous_time_exact_sol_1(viscosity, nb_x_data, x_data, nb_t_data, t_data):

 # viscosity parameter
 nu    = viscosity

 # order of quadrature rule
 qn    = 64
 qx,qw = hermgauss(qn)

 # compute solution u(x,t) by quadrature of analytical formula:
 u_quad = np.zeros([nb_x_data,nb_t_data])

 for utj in range(nb_t_data):

     if (t_data[utj]==0.0):
         for uxj in range(nb_x_data):
             u_quad[uxj,utj] = -np.sin(np.pi*x_data[uxj])

     else:
         for uxj in range(nb_x_data):
             top = 0.0
             bot = 0.0
             for qj in range(qn):
                 c   = 2.0*np.sqrt(nu*t_data[utj])
                 top = top - qw[qj]*c*np.sin(np.pi*(x_data[uxj]-c*qx[qj]))*np.exp(-np.cos(np.pi*(x_data[uxj]-c*qx[qj]))/(2.0*np.pi*nu))
                 bot = bot + qw[qj]*c*np.exp(-np.cos(np.pi*(x_data[uxj]-c*qx[qj]))/(2.0*np.pi*nu))

                 u_quad[uxj,utj] = top/bot

 return u_quad


