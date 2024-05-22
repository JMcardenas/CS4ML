# Import packages
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, BatchNormalization, Dropout, Input
from keras import optimizers

import tensorflow as tf
import keras.backend as K

import time, os, argparse, io, shutil, sys, math, socket, random, hdf5storage

import numpy as np
import scipy.io as sio

#Clean previous sessions
tf.keras.backend.clear_session() 

# Import functions
from main_functions import init_model, extract_model, get_M_and_k_values, evaluate_data_model
from main_functions import get_measurement_matrix, get_numerical_dim, get_k_ratios, CAS_method, MC_method
from Burgers_PDE_1D import fun_u_0, fun_u_b, fun_r, burgers_viscous_time_exact_sol_1

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step(X_r, X_data, u_data, weights_r, weights_data):
     
    DTYPE='float32'
    # Set constants
    pi = tf.constant(np.pi, dtype=DTYPE)
    viscosity = .01/pi
    
    # Compute loss function and gradients
    with tf.GradientTape(persistent=True) as tape:

        # This tape is for derivatives wrt trainable variables
        tape.watch(model.trainable_variables)

        # Compute the PDE
        with tf.GradientTape(persistent=True) as tape2:
            # Split t and x to compute partial derivatives
            t, x = X_r[:, 0:1], X_r[:,1:2]

            # watch twice variable x
            tape2.watch(x)

            # Nested Gradicent tape for 2nd derivative
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x)
                tape1.watch(t)

                # evaluate model 
                u = model(tf.stack([t[:,0], x[:,0]], axis=1))

            # Compute u_t & u_x
            u_x = tape1.gradient(u, x)   
            u_t = tape1.gradient(u, t)
        
        # Compute u_xx
        u_xx = tape2.gradient(u_x, x)
        
        # get residual PDE
        residual = u_t + u * u_x - viscosity * u_xx        
        
        # Compute loss function
        if len(weights_r)>0:
            phi_r = tf.reduce_mean(tf.matmul(weights_r,tf.square(residual)))
        else:  
            phi_r = tf.reduce_mean(tf.square(residual))
        
        # Initialize loss
        loss = phi_r
        
        # Add inital and boundary loss
        for i in range(len(X_data)):
            # model on data
            u_pred = model(X_data[i])

            if len(weights_data)>0:
                loss += tf.reduce_mean(tf.matmul(weights_data[i],tf.square(u_data[i] - u_pred)))
            else:
                loss += tf.reduce_mean(tf.square(u_data[i] - u_pred))         
    
    # get gradient
    grad_theta = tape.gradient(loss, model.trainable_variables)
           
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))

    return loss

if __name__ == '__main__': 

    print('Running tensorflow with version:')
    print(tf.__version__)

    # depending on where running, change scratch/project directories
    # TODO: change these when installed on your local machine!!!
    
    scratchdir = '/content/'
    projectdir = '/content/'

    print(scratchdir)

    timestamp  = str(int(time.time()));
    start_time = time.time()

    # parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_opt", default= 1, type= int, help="training option, if 1 then train, if 0 then test model")
    parser.add_argument("--nb_layers", default = 5, type = int, help = "Number of hidden layers")
    parser.add_argument("--nb_nodes_per_layer", default = 50, type = int, help = "Number of nodes per hidden layer")
    parser.add_argument("--nb_train_points", default = 16000, type = int, help = "Number of points to use in training")
    parser.add_argument("--train_pointset", default = 'uniform_random', type = str, help = "Type of points to use in training")
    parser.add_argument("--nb_test_points", default = 3937, type = int, help = "Number of points to use in testing")
    parser.add_argument("--test_pointset", default = 'CC_sparse_grid', type = str, help = "Type of points to use in testing")
    parser.add_argument("--nb_epochs", default = 10000, type = int, help = "Number of epochs for training")
    parser.add_argument("--batch_size", default = 1000, type = int, help = "Number of training samples per batch")
    parser.add_argument("--nb_trials", default = 1, type = int, help = "Number of trials to run for averaging results")
    parser.add_argument("--train", default = 0, type = int, help = "Switch for training or testing")
    parser.add_argument("--make_plots", default = 0, type = int, help = "Switch for making plots")
    parser.add_argument("--run_ID", type = str, help = "String for naming batch of trials in this run")
    parser.add_argument("--blocktype", default = 'default', type = str, help = "Type of building block for hidden layers, e.g., ResNet vs. default")
    parser.add_argument("--activation", default = 'tanh', type = str, help = "Type of activation function to use")
    parser.add_argument("--example", type = int, help = "Example function to approximate (a number 1-2)")
    parser.add_argument("--optimizer", default = 'Adam', type = str, help = "Optimizer to use in minimizing the loss")
    parser.add_argument("--initializer", default = 'normal', type = str, help = "Initializer to use for the weights and biases")
    parser.add_argument("--quiet", default = 0, type = int, help = "Switch for verbose output")
    parser.add_argument("--input_dim", default = 2, type = int, help = "Dimension of the input")
    parser.add_argument("--output_dim", default = 1, type = int, help = "Dimension of the output")
    parser.add_argument("--MATLAB_data", default = 1, type = int, help = "Switch for using MATLAB input data points")
    parser.add_argument("--trial_num", default = 0, type = int, help = "Number for the trial to run")
    parser.add_argument("--precision", default = 'single', type = str, help = "Switch for double vs. single precision")
    parser.add_argument("--use_regularizer", default = 0, type = int, help = "Switch for using regularizer")
    parser.add_argument("--SG_level", default = 1, type = int, help = "The level of the sparse grid rule for testing")
    parser.add_argument("--reg_lambda", default = "1e-3", type = str, help = "Regularization parameter lambda")
    parser.add_argument("--loss_function", default = "MSE", type = str, help = "Loss function to minimize with optimizer set by arguments")
    parser.add_argument("--error_tol", default = "5e-7", type = str, help = "Stopping tolerance for the solvers")
    parser.add_argument("--sigma", default = "1e-1", type = str, help = "Standard deviation for normal initializer, max and min for uniform symmetric initializer, constant for constant initializer")
    parser.add_argument("--training_method", default = "MC", type = str, help = "Method to use for training/sampling strategy (MC = Monte Carlo sampling with retraining, CAS = Christoffel Adaptive sampling with retraining")
    parser.add_argument("--training_steps", default = 10, type = int, help = "Number of steps to use if training with a multi-step procedure (e.g., 10 steps)")
    parser.add_argument("--nb_epochs_per_iter", default = 5000, type = int, help = "Number of epochs per iteration of multi-step procedures")
    parser.add_argument("--nb_schedule_epochs", default = "constant", type = str, help = "Type of schedule epochs (constant = increase a fix nb_epochs, variable = inscrease iter times nb_init_epochs per iteration)")
    parser.add_argument("--lrn_rate_schedule", default = "exp_decay", type = str, help = "Learning rate schedule, e.g. exp_decay")
    args = parser.parse_args()

    print('using ' + args.optimizer + ' optimizer')

    if args.train:
        print('batching with ' + str(args.batch_size) + ' out of ' + 
               str(args.nb_train_points) + ' ' + args.train_pointset + 
               ' training points')

    # set the standard deviation for initializing the DNN weights and biases
    if args.initializer == 'normal' or args.initializer == 'he_normal':
        sigma = float(args.sigma)
        print('initializing (W,b) with N(0, ' + str(sigma) + '^2)')

    elif args.initializer == 'uniform':
        sigma = float(args.sigma)
        print('initializing (W,b) with U(-' + str(sigma) + ', ' + str(sigma) + ')')

    elif args.initializer == 'constant':
        sigma = float(args.sigma)
        print('initializing (W,b) as constant ' + str(sigma))

    else: 
        print('incorrect initializer: use normal, he_normal, uniform, or constant')

    # set the precision variable to initialize weights and biases in either double or single precision
    if args.precision == 'double':
        print('Using double precision') 
        precision = tf.float64
        error_tol = float(args.error_tol)
        tf.keras.backend.set_floatx('float64')
 
    elif args.precision == 'single':
        print('Using single precision')
        precision = tf.float32
        error_tol = float(args.error_tol)
        tf.keras.backend.set_floatx('float32')

    else:
        print('incorrect precision: use double or single')

    # set the unique run ID used in many places, e.g., directory names for output
    if args.run_ID is None:
        unique_run_ID = timestamp
    else:
        unique_run_ID = args.run_ID

    # set the seeds for numpy and tensorflow to ensure all initializations are the same
    np_seed = 0
    tf_seed = 0

    # record the trial number
    trial = args.trial_num

    # unique key for naming results
    key = unique_run_ID + '/' + args.activation + '_' + args.blocktype + '_' + str(args.nb_layers) + 'x' + \
            str(args.nb_nodes_per_layer) + '_' + str(args.nb_train_points).zfill(6) +'_pnts_' + \
            str(error_tol) + '_tol_' + args.optimizer + '_opt_' + args.nb_schedule_epochs + '_schedule_epochs' +\
            '_burgers_example_' + str(args.example) + '_dim_' + str(args.input_dim) +\
            '_training_method_' + str(args.training_method)

    print('using key:', key)

    # the results and scratch directory can be individually specified (however for now they are the same)
    result_folder  = scratchdir + '/colab_results/' + key + '/trial_' + str(trial)
    scratch_folder = scratchdir + '/colab_results/' + key + '/trial_' + str(trial)

    # create the result folder if it doesn't exist yet
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # create the scratch folder if it doesn't exist yet
    if not os.path.exists(scratch_folder):
        os.makedirs(scratch_folder)

    print('Saving results to:', result_folder)

    # loading the training data from MATLAB 
    if args.MATLAB_data:

        # the training data is in MATLAB files with names in the form:  
        # training_data_example_(nb_example)_dim_(input_dim).mat
        training_data_filename = scratchdir + '/colab_data' + '/train_data_example_' + \
                                 str(args.example) + '_dim_' + str(args.input_dim) + '.mat' 

        print('Loading training data from: ' + training_data_filename) 

        # load the MATLAB -v7.3 hdf5-based format mat file 
        training_data = hdf5storage.loadmat(training_data_filename) 

        # number of points 
        N_r = (training_data['nb_x_col_data'][0][0]).astype(int)
        N_b = (training_data['nb_bound_data'][0][0]).astype(int)
        N_0 = (training_data['nb_init_data'][0][0]).astype(int)

        # collocation data
        x_c = tf.convert_to_tensor(training_data['x_col_data'], dtype= precision)
        t_c = tf.convert_to_tensor(training_data['t_col_data'], dtype= precision)
        u_c = tf.convert_to_tensor(training_data['u_col_data'], dtype= precision)
        X_r = tf.concat([t_c, x_c], axis= 1)

        # inital data 
        t_0 = tf.convert_to_tensor(training_data['t_init_data'], dtype= precision)
        x_0 = tf.convert_to_tensor(training_data['x_init_data'], dtype= precision)
        u_0 = tf.convert_to_tensor(training_data['u_init_data'], dtype= precision)
        X_0 = tf.concat([t_0, x_0], axis= 1)

        # boundary data 
        t_b = tf.convert_to_tensor(training_data['t_bound_data'], dtype= precision)
        x_b = tf.convert_to_tensor(training_data['x_bound_data'], dtype= precision)
        u_b = tf.convert_to_tensor(training_data['u_bound_data'], dtype= precision)
        X_b = tf.concat([t_b, x_b], axis= 1)       

        # Collect boundary and inital data in lists
        X_data = [X_0, X_b]
        u_data = [u_0, u_b]
        
        # collect collocation, inital and boundary data in list
        x_all_data = tf.concat([x_c, x_0, x_b], axis= 0)
        t_all_data = tf.concat([t_c, t_0, t_b], axis= 0)
        u_all_data = tf.concat([u_c, u_0, u_b], axis= 0)

        # Set up train data 
        X_train_data = tf.concat([t_all_data, x_all_data], axis= 1)
        u_train_data = u_all_data    

    else:
        # Set number of data points
        N_0 = 1000
        N_b = 1000
        N_r = 20000

        # Set boundary
        tmin = 0.
        tmax = 1.
        xmin = -1.
        xmax = 1.

        # Lower bounds
        lb = tf.constant([tmin, xmin], dtype= precision)
        # Upper bounds
        ub = tf.constant([tmax, xmax], dtype= precision)

        # Set random seed for reproducible results
        tf.random.set_seed(0)

        # Draw uniform sample points for initial boundary data
        t_0 = tf.ones((N_0,1), dtype= precision)*lb[0]
        x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype= precision)
        X_0 = tf.concat([t_0, x_0], axis=1)

        # Evaluate intitial condition at x_0
        u_0 = fun_u_0(x_0)

        # Boundary data
        t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype= precision) 
        x_b = tf.random.uniform((N_b,1), lb[1], ub[1], dtype= precision)
        x_b_np = np.zeros((x_b.shape[0],1))

        for i in range(x_b.shape[0]):
            if x_b[i] <= 0:
                x_b_np[i] = -1
            else: 
                x_b_np[i] = 1

        x_b = tf.convert_to_tensor(x_b_np, dtype= precision)
        X_b = tf.concat([t_b, x_b], axis=1)

        # Evaluate boundary condition at (t_b,x_b)
        u_b = fun_u_b(t_b, x_b)

        # Draw uniformly sampled collocation points
        t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype= precision)
        x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype= precision)
        X_r = tf.concat([t_r, x_r], axis=1)

        # Collect boundary and inital data in lists
        X_data = [X_0, X_b]
        u_data = [u_0, u_b]

        print('-----------------------------------------------------------------------')
        print('size collocation data: ', X_r.shape)
        print('size inital data: ', X_0.shape)
        print('size boundary data: ', X_b.shape)
        print('-----------------------------------------------------------------------')

    #load testing data    
    # the testing data is in MATLAB files with names in the form:  
    # test_data_example_(nb_example)_dim_(input_dim).mat
    testing_data_filename = scratchdir + '/colab_data' + '/test_data_example_' + \
                            str(args.example) + '_dim_' + str(args.input_dim) + '.mat' 

    print('Loading testing data from: ' + testing_data_filename) 

    # load the MATLAB -v7.3 hdf5-based format mat file 
    testing_data = hdf5storage.loadmat(testing_data_filename)
    
    x_test_data  = testing_data['x_data']
    t_test_data  = testing_data['t_data']
    u_test_data  = testing_data['u_data']
    nb_test_pts  = testing_data['nb_pts'][0][0]

    # Set up meshgrig 
    X_test_grid = np.vstack([t_test_data.flatten(), x_test_data.flatten()]).T
    u_real_test = u_test_data.reshape(nb_test_pts,1)

    print('-------------------------------------------------------------------')
    print('size x_test_data: ', x_test_data.shape)
    print('size t_test_data: ', t_test_data.shape)
    print('size u_test_data: ', u_test_data.shape)
    print('size X_test_grid: ', X_test_grid.shape)
    print('size u_real: '     , u_real_test.shape)
    print('-------------------------------------------------------------------')
    
    #--------------------------------------------------------------------------#
    # load plot data 

    plot_data_filename = scratchdir + '/colab_data' + '/plot_data_example_' + \
                         str(args.example) + '_dim_' + str(args.input_dim) + '.mat'  

    print('Loading plot data from: ' + plot_data_filename) 

    plot_data = hdf5storage.loadmat(plot_data_filename)

    Z_plot_grid = tf.convert_to_tensor(plot_data['Z_grid'], dtype= precision)
    u_plot_data = tf.convert_to_tensor(plot_data['u_plot_data'], dtype= precision) 

    # number of points 
    nb_plot_col_data   = (plot_data['nb_plot_col_data'][0][0]).astype(int)
    nb_plot_init_data  = (plot_data['nb_plot_init_data'][0][0]).astype(int)
    nb_plot_bound_data = (plot_data['nb_plot_bound_data'][0][0]).astype(int)
    nb_plot_x_data     = (plot_data['nb_plot_x_data'][0][0]).astype(int)
    nb_plot_t_data     = (plot_data['nb_plot_t_data'][0][0]).astype(int)

    # collocation data
    x_plot_col_data = tf.convert_to_tensor(plot_data['x_plot_col_data'], dtype= precision)
    t_plot_col_data = tf.convert_to_tensor(plot_data['t_plot_col_data'], dtype= precision)
    u_plot_col_data = tf.convert_to_tensor(plot_data['u_plot_col_data'], dtype= precision)
    Z_plot_col_data = tf.concat([t_plot_col_data, x_plot_col_data], axis= 1)

    # inital data 
    x_plot_init_data = tf.convert_to_tensor(plot_data['x_plot_init_data'], dtype= precision)
    t_plot_init_data = tf.convert_to_tensor(plot_data['t_plot_init_data'], dtype= precision)
    u_plot_init_data = tf.convert_to_tensor(plot_data['u_plot_init_data'], dtype= precision)
    Z_plot_init_data = tf.concat([t_plot_init_data, x_plot_init_data], axis= 1)

    # boundary data   
    x_plot_bound_data = tf.convert_to_tensor(plot_data['x_plot_bound_data'], dtype= precision)
    t_plot_bound_data = tf.convert_to_tensor(plot_data['t_plot_bound_data'], dtype= precision)
    u_plot_bound_data = tf.convert_to_tensor(plot_data['u_plot_bound_data'], dtype= precision)
    Z_plot_bound_data = tf.concat([t_plot_bound_data, x_plot_bound_data], axis= 1)


    if args.training_opt == 1: 
        print('Training option:', str(args.training_opt)) 
        #--------------------------------------------------------------------------#
        # If train_opt true, then train the model 
        #--------------------------------------------------------------------------#
        # SETUP RUN DATA 
    
        # set schedule epochs
        if args.nb_schedule_epochs == 'constant':
            nb_epochs_vec   = args.nb_epochs_per_iter*np.ones((1,args.training_steps-1))[0].astype(int)
            nb_total_epochs = int(np.sum(nb_epochs_vec)) 
    
        elif args.nb_schedule_epochs == 'doublein':
            nb_epochs_inc   = np.linspace(1,args.training_steps-1,args.training_steps-1).astype(int)
            nb_epochs_vec   = args.nb_epochs_per_iter*nb_epochs_inc
            nb_total_epochs = int(np.sum(nb_epochs_vec)) 
    
        elif args.nb_schedule_epochs == 'midlein':
            nb_epochs_inc = np.array([])
            for i in range(args.training_steps-1):
                    nb_epoch_i    = (i+2)/2
                    nb_epochs_inc = np.append(nb_epochs_inc, nb_epoch_i) 
    
            nb_epochs_vec   = (args.nb_epochs_per_iter*nb_epochs_inc).astype(int)
            nb_total_epochs = int(np.sum(nb_epochs_vec))
    
        else:
            print('incorrect nb_schedule_epochs: use constant, midlein, or doublein')
        
        print('nb schedule epochs:  ', args.nb_schedule_epochs)
        print('nb of epochs vector: ', nb_epochs_vec)
        print('nb of total epochs:  ', str(nb_total_epochs))
        
        # weighted opt
        if args.training_method == 'CAS':
            weights_opt = True
        else:
            weights_opt = False
        
        # set number of pnts
        N_max              = args.nb_nodes_per_layer
        nb_train_col_pts   = N_r
        nb_train_init_pts  = N_0
        nb_train_bound_pts = N_b
        
        # Chose samples values and sampling ratios
        M_col_values, k_col_values     = get_M_and_k_values(args.nb_nodes_per_layer, args.input_dim, nb_train_col_pts, args.training_steps, N_max, 'collocation')
        M_init_values, k_init_values   = get_M_and_k_values(args.nb_nodes_per_layer, args.input_dim, nb_train_init_pts, args.training_steps, N_max, 'initial')
        M_bound_values, k_bound_values = get_M_and_k_values(args.nb_nodes_per_layer, args.input_dim, nb_train_bound_pts, args.training_steps, N_max, 'boundary')
    
        print('-----------------------------------------------------------------------')
        print('k col vals: ' + str(k_col_values) + ' | k init vals: '+ str(k_init_values) + ' | k bound vals: ' + str(k_bound_values))
        print('M col vals: ' + str(M_col_values) + ' | M init vals: '+ str(M_init_values) + ' | M bound vals: ' + str(M_bound_values))
        print('-----------------------------------------------------------------------')        
        
        #--------------------------------------------------------------------------#
        # Initialize model aka u_\theta
        #--------------------------------------------------------------------------#
        model = init_model(args.nb_layers, args.nb_nodes_per_layer, args.input_dim, args.output_dim, args.activation, args.initializer, args.trial_num)
        
        # number of variables
        model_num_trainable_variables = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    
        if not args.quiet:
            print('This model has {} trainable variables'.format(model_num_trainable_variables))
            model.summary()
    
        #--------------------------------------------------------------------------#
        # set up learning rate schedule from either exp_decay, linear, or constant
        #--------------------------------------------------------------------------#
    
        if args.lrn_rate_schedule == 'exp_decay':
            init_rate   = 1e-3
            lrn_rate    = init_rate
            decay_steps = 1e3
    
            final_learning_rate = error_tol
            
            base = np.exp(decay_steps/nb_total_epochs*(np.log(final_learning_rate)-np.log(init_rate))) 
    
            print('based on init rate = ' + str(init_rate)
                + ', decay_steps = ' + str(decay_steps)
                + ', calculated base = ' + str(base)
                + ', so that after ' + str(nb_total_epochs)
                + ' epochs we have final learning rate = '
                + str(init_rate*base**(nb_total_epochs/decay_steps)))
            
            decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                init_rate, decay_steps, base, staircase= False, name= None)
    
        elif args.lrn_rate_schedule == 'constant':
            # We choose a piecewise decay of the learning rate, i.e., the
            # step size in the gradient descent type algorithm
            # the first 1000 steps use a learning rate of 0.01
            # from 1000 - 3000: learning rate = 0.001
            # from 3000 onwards: learning rate = 0.0005
            decay_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
    
        else: 
            print('incorrect learning rate schedule: use exp_decay or constant')
    
        #--------------------------------------------------------------------------#
        # set up optimizers for training
        #--------------------------------------------------------------------------#
    
        if args.optimizer == 'SGD':
            optim = tf.keras.optimizers.SGD(
                learning_rate= 1e-4,
                name= 'SGD'
            )
        elif args.optimizer == 'Adam': 
            optim = tf.keras.optimizers.Adam(
                learning_rate= decay_schedule,
                beta_1= 0.9, beta_2= 0.999, epsilon= 1e-07,
                name = 'Adam'
            )
        elif args.optimizer == 'LBDF':
            print('not implemented yet ')
    
        else: 
            print('optimizer must be one of the present optimizers: SGD or Adam')
    
        #--------------------------------------------------------------------------#
        run_data                       = {}
        run_data['trial']              = args.trial_num
        run_data['key']                = key
        run_data['precision']          = args.precision
    
        run_data['nb_layers']          = args.nb_layers
        run_data['nb_nodes_per_layer'] = args.nb_nodes_per_layer
        run_data['activation']         = args.activation
        run_data['blocktype']          = args.blocktype
        run_data['initializer']        = args.initializer
    
        run_data['loss_function']      = args.loss_function
        run_data['optimizer']          = args.optimizer
        run_data['lrn_rate_schedule']  = args.lrn_rate_schedule
        run_data['error_tol']          = args.error_tol
        run_data['sigma']              = args.sigma
        run_data['nb_schedule_epochs'] = args.nb_schedule_epochs
        run_data['nb_epochs_per_iter'] = args.nb_epochs_per_iter
    
        run_data['training_method']    = args.training_method 
        run_data['training_steps']     = args.training_steps
        run_data['tf_trainable_vars']  = model_num_trainable_variables
    
        run_data['M_col_values']       = M_col_values
        run_data['M_init_values']      = M_init_values
        run_data['M_bound_values']     = M_bound_values   
        run_data['k_col_values']       = k_col_values
        run_data['k_init_values']      = k_init_values
        run_data['k_bound_values']     = k_bound_values
        
        run_data['result_folder']      = result_folder
        run_data['model_save_folder']  = 'final'
        run_data['run_data_filename']  = 'run_data.mat'
        
        hist          = []
        I_c_grid      = np.arange(0,nb_train_col_pts)
        I_0_grid      = np.arange(0,nb_train_init_pts)
        I_b_grid      = np.arange(0,nb_train_bound_pts)
        I_c_ad        = np.array([])
        I_0_ad        = np.array([])
        I_b_ad        = np.array([])
        I_c           = np.array([])
        I_0           = np.array([])
        I_b           = np.array([])
        I_c_new       = np.array([])
        I_0_new       = np.array([])
        I_b_new       = np.array([])
        r_col_vals    = np.array([])
        r_init_vals   = np.array([])
        r_bound_vals  = np.array([])
    
        l2_error_u_train      = np.array([])
        l2_error_data_u_train = np.array([])
        l2_error_u_test       = np.array([])
        l2_error_data_u_test  = np.array([])    
    
        #------------------------------------------------------------------------#
        print('-----------------------------------------------------------------------')
        print('Training DNN: ' + str(key)) 
        
        # time iterations 
        t1 = time.time()
    
        for iter in range(args.training_steps-1):
             
            # Set the number of epochs
            if iter == 0:
                nb_epochs_last_iter = 0
            else:
                nb_epochs_last_iter = last_epoch_from_training
    
            nb_epochs = nb_epochs_vec[iter] + nb_epochs_last_iter
    
            print('===================================================================')
            print('| Trial:', trial,' | Training iteration: ', iter)
    
            #1.1. Collocation data: compute B_model, B_matrix, and numerical dimension
            if iter == 0:
                B_model             = extract_model(model) 
                B_col_matrix        = evaluate_data_model(B_model, X_r)/np.sqrt(nb_train_col_pts) 
                U_col_matrix, r_col = get_numerical_dim(B_col_matrix, args.nb_nodes_per_layer)
            else:
                U_col_matrix = Phi_col_basis
                r_col        = r_col_Phi
            
            print('-------------------------------------------------------------------')
            print('numerical dimension on collocation data: ', r_col)
    
            r_col_vals = np.append(r_col_vals, [r_col])
            
            #1.2. Compute numerical dimension on initial data
            if iter == 0:
                B_init_matrix         = evaluate_data_model(B_model, X_0)/np.sqrt(nb_train_init_pts) 
                U_init_matrix, r_init = get_numerical_dim(B_init_matrix, args.nb_nodes_per_layer)
            else:
                U_init_matrix = Phi_init_basis
                r_init        = r_init_Phi
    
            print('numerical dimension on inital data:      ', r_init)
    
            r_init_vals = np.append(r_init_vals, [r_init])
    
            #1.3. Compute numerical dimension on boundary data
            if iter == 0:
                B_bound_matrix          = evaluate_data_model(B_model, X_b)/np.sqrt(nb_train_bound_pts) 
                U_bound_matrix, r_bound = get_numerical_dim(B_bound_matrix, args.nb_nodes_per_layer)
            else:
                U_matrix_bound = Phi_bound_basis
                r_bound        = r_bound_Phi            
    
            print('numerical dimension on boundary data:    ', r_bound)
    
            r_bound_vals = np.append(r_bound_vals, [r_bound])
            
            #2. Sampling strategy 
            if args.training_method == 'CAS':   
                #2.1 Collocation points
                # get k ratios
                k_max_ratio, k_min_ratio, s_l = get_k_ratios(M_col_values, k_col_values, args.nb_nodes_per_layer, r_col, iter)
                # draw samples 
                I_c, weights_c = CAS_method(U_col_matrix, r_col, s_l, k_min_ratio, k_max_ratio, nb_train_col_pts, args.nb_nodes_per_layer, iter, I_c_grid, I_c_new, I_c_ad, I_c, weights_opt)
    
                #2.2 Inital points 
                # get k ratios
                k_0_max_ratio, k_0_min_ratio, s_0_l = get_k_ratios(M_init_values, k_init_values, args.nb_nodes_per_layer, r_init, iter)
                # draw samples 
                I_0, weights_0 = CAS_method(U_init_matrix, r_init, s_0_l, k_0_min_ratio, k_0_max_ratio, nb_train_init_pts, args.nb_nodes_per_layer, iter, I_0_grid, I_0_new, I_0_ad, I_0, weights_opt)
    
                #2.3 Boundary points
                # get k ratios
                k_b_max_ratio, k_b_min_ratio, s_b_l = get_k_ratios(M_bound_values, k_bound_values, args.nb_nodes_per_layer, r_bound, iter)
                # draw samples 
                I_b, weights_b = CAS_method(U_bound_matrix, r_bound, s_b_l, k_b_min_ratio, k_b_max_ratio, nb_train_bound_pts, args.nb_nodes_per_layer, iter, I_b_grid, I_b_new, I_b_ad, I_b, weights_opt)
    
            elif args.training_method == 'MC':
                #2.1 Collocation points
                I_c, weights = MC_method(M_col_values, iter, I_c)
    
                #2.2 Inital points
                I_0, weights = MC_method(M_init_values, iter, I_0)
    
                #2.3 Boundary points
                I_b, weights = MC_method(M_bound_values, iter, I_b)
    
            else: 
                print('Incorrect sampling method')          
    
            print('-------------------------------------------------------------------')
            print('It pts: '   + str(I_c.shape[0]+I_0.shape[0]+I_b.shape[0]) +\
                ' | I_c set: ' + str(I_c.shape[0]) +\
                ' | I_0 set: ' + str(I_0.shape[0]) +\
                ' | I_b set: ' + str(I_b.shape[0])) 
            
            #3. Extract collocation data 
            X_col   = tf.gather(X_r, indices= I_c)
            u_col   = tf.gather(u_c, indices= I_c)
    
            #4. Collect boundary and inital data in lists
            X_init  = tf.gather(X_0, indices= I_0)
            u_init  = tf.gather(u_0, indices= I_0)
    
            X_bound = tf.gather(X_b, indices= I_b)
            u_bound = tf.gather(u_b, indices= I_b)
    
            X_init_bound_data = [X_init, X_bound]
            u_init_bound_data = [u_init, u_bound] 
    
            if weights_opt: 
                weights_r    = tf.convert_to_tensor(np.asmatrix(np.diag(weights_c)), dtype= precision)
                weights_0    = tf.convert_to_tensor(np.asmatrix(np.diag(weights_0)), dtype= precision)
                weights_b    = tf.convert_to_tensor(np.asmatrix(np.diag(weights_b)), dtype= precision)
                weights_data = [weights_0, weights_b]
            else: 
                weights_r    = []
                weights_data = []          
                
            #----------------------------------------------------------------------#
            # Training 
            #----------------------------------------------------------------------#                                         
            for i in range(nb_epochs_last_iter, nb_epochs + 1):
                    
                # train step 
                loss = train_step(X_col, X_init_bound_data, u_init_bound_data, weights_r, weights_data) 
    
                # Append current loss to hist
                hist.append(loss.numpy())
                
                # update nb epoch  
                last_epoch_from_training = i
                
                if (i == nb_epochs) or ((iter == 0) and (i == 0)):  
    
                    # Report error over Training data 
                    X_current_train_data = tf.concat([X_col, X_init, X_bound], axis=0)
                    u_current_train_data = tf.concat([u_col, u_init, u_bound], axis=0)
                    
                    u_pred_train = (evaluate_data_model(model, X_current_train_data)).numpy() 
    
                    u_train_norm          = np.sqrt(np.sum(np.square(np.abs(u_current_train_data))))
                    abs_train_error_norm  = np.sqrt(np.sum(np.square(np.abs(u_pred_train - u_current_train_data))))
    
                    l2_error_u_train      = abs_train_error_norm/u_train_norm 
                    l2_error_data_u_train = np.append(l2_error_data_u_train, l2_error_u_train)
                    
                    # Report error over Testing data 
                    u_pred_test = (evaluate_data_model(model, X_test_grid)).numpy() 
    
                    u_test_norm          = np.sqrt(np.sum(np.square(np.abs(u_test_data))))
                    abs_test_error_norm  = np.sqrt(np.sum(np.square(np.abs(u_pred_test - u_real_test))))
                    l2_error_u_test      = abs_test_error_norm/u_test_norm 
                    l2_error_data_u_test = np.append(l2_error_data_u_test, l2_error_u_test)
    
                    # Print loss
                    print('-------------------------------------------------------------------') 
                    print('It {:05d}: l2-Training loss = {:10.8e}'.format(i,l2_error_u_train))
                    print('It {:05d}: l2-Testing loss  = {:10.8e}'.format(i,l2_error_u_test))
                
            #5. Compute basis, numerical dim for next iter, and Christoffel function
            if iter < args.training_steps:
                #5.1 Collocation points  
                B_col_matrix                 = get_measurement_matrix(model, X_r, nb_train_col_pts)  
                Phi_col_basis, r_col_Phi     = get_numerical_dim(B_col_matrix, args.nb_nodes_per_layer)
                
                #5.2 Inital points  
                B_init_matrix                = get_measurement_matrix(model, X_0, nb_train_init_pts)  
                Phi_init_basis, r_init_Phi   = get_numerical_dim(B_init_matrix, args.nb_nodes_per_layer)
        
                #5.3 Boundary points  
                B_bound_matrix               = get_measurement_matrix(model, X_b, nb_train_bound_pts)    
                Phi_bound_basis, r_bound_Phi = get_numerical_dim(B_bound_matrix, args.nb_nodes_per_layer)
        
                print('-------------------------------------------------------------------')
                print('Phi r-value: ',' col data:',r_col_Phi,' init data:',r_init_Phi,' bound data:', r_bound_Phi) 

        #--------------------------------------------------------------------------#
        # Save data
        #--------------------------------------------------------------------------# 
        run_data['col_samples']       = tf.gather(X_r, indices= I_c).numpy()
        run_data['init_samples']      = tf.gather(X_0, indices= I_0).numpy()
        run_data['bound_samples']     = tf.gather(X_b, indices= I_b).numpy()
        
        run_data['u_0_method']        = tf.gather(u_0, indices= I_0)[:,0:1].numpy()
        run_data['u_b_method']        = tf.gather(u_b, indices= I_b)[:,0:1].numpy()
    
        run_data['nb_epochs_vec']     = nb_epochs_vec
        run_data['nb_total_epochs']   = nb_total_epochs
        run_data['nb_epochs']         = nb_epochs
        
        run_data['hist_loss']         = hist
        run_data['l2_error_u_test']   = l2_error_data_u_test   
        run_data['l2_error_u_train']  = l2_error_data_u_train 
        
        run_data['u_pred']            = u_pred_test
        
        run_data['r_col_vals']        = r_col_vals
        run_data['r_init_vals']       = r_init_vals
        run_data['r_bound_vals']      = r_bound_vals

        # save model 
        model.save(run_data['result_folder'] + '/' + run_data['model_save_folder'])
        print('model save to: ', run_data['result_folder'] + '/' + run_data['model_save_folder'])
        
        # save the resulting mat file with scipy.io
        sio.savemat(run_data['result_folder'] + '/' + run_data['run_data_filename'], run_data) 
        print('save to:',run_data['result_folder'] + '/' + run_data['run_data_filename'])
        
        print('\nComputation time total: {} minutes'.format((time.time()-t1)/60))
        #----------------------------------------------------------------------#
    else: 
        t1 = time.time()
        #----------------------------------------------------------------------#
        #If not doing training, test the model
        #----------------------------------------------------------------------#
        print('Training option:', str(args.training_opt), 'Then Testing model from:') 
        # open result for each trial
        for trial in range(args.nb_trials):

            # the result and scratch folder (here are the same since we save to 
            # scratch only, change if needed) 

            result_folder  = scratchdir + '/colab_results/' + key + '/trial_' + str(trial)
            scratch_folder = scratchdir + '/colab_results/' + key + '/trial_' + str(trial)

            if not args.quiet:
                print("Loading run \"%s\" trial: %d from %s" % (unique_run_ID, trial, result_folder))
 
            # load model 
            model = tf.keras.models.load_model(result_folder + '/final')
            
            model_num_trainable_variables = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
            
            if not args.quiet:
                print('This standard model has ' + str(model_num_trainable_variables) + ' trainable parameters')
           
            # Compute data for plot    
            model_plot_data = (model(Z_plot_grid)).numpy().reshape(nb_plot_x_data, nb_plot_t_data)

            # Compute basis, numerical dim on plot data 
            # Collocation data
            B_plot_col_matrix                      = get_measurement_matrix(model, Z_plot_col_data , nb_plot_col_data)  
            Phi_plot_col_basis, r_plot_col_Phi     = get_numerical_dim(B_plot_col_matrix, args.nb_nodes_per_layer)
            
            # Inital data
            B_plot_init_matrix                     = get_measurement_matrix(model, Z_plot_init_data, nb_plot_init_data)  
            Phi_plot_init_basis, r_plot_init_Phi   = get_numerical_dim(B_plot_init_matrix, args.nb_nodes_per_layer)
        
            # Boundary data  
            B_plot_bound_matrix                    = get_measurement_matrix(model, Z_plot_bound_data, nb_plot_bound_data)    
            Phi_plot_bound_basis, r_plot_bound_Phi = get_numerical_dim(B_plot_bound_matrix, args.nb_nodes_per_layer)
        
            print('-------------------------------------------------------------------')
            print('Phi plot r-value: ',' col data:',r_plot_col_Phi,' init data:',r_plot_init_Phi,' bound data:', r_plot_bound_Phi) 

            #--------------------------------------------------------------------------#
            # compute CF and PD on plot data
            Phi_plot_col_matrix   = np.asmatrix(Phi_plot_col_basis)[:,0:r_plot_col_Phi]
            Prob_plot_col_dist    = np.sum(np.square(np.abs(Phi_plot_col_matrix)), axis= 1)/r_plot_col_Phi
            Chris_plot_col_fun    = np.divide(1, Prob_plot_col_dist)
        
            Phi_plot_init_matrix  = np.asmatrix(Phi_plot_init_basis)[:,0:r_plot_init_Phi]
            Prob_plot_init_dist   = np.sum(np.square(np.abs(Phi_plot_init_matrix)), axis= 1)/r_plot_init_Phi
            Chris_plot_init_fun   = np.divide(1, Prob_plot_init_dist)
        
            Phi_plot_bound_matrix = np.asmatrix(Phi_plot_bound_basis)[:,0:r_plot_bound_Phi]
            Prob_plot_bound_dist  = np.sum(np.square(np.abs(Phi_plot_bound_matrix)), axis= 1)/r_plot_bound_Phi
            Chris_plot_bound_fun  = np.divide(1, Prob_plot_bound_dist)
        
            print('-------------------------------------------------------------------')
            print('Christoffel plot col function size:   ', Chris_plot_col_fun.shape)
            print('Christoffel plot init function size:  ', Chris_plot_init_fun.shape)
            print('Christoffel plot bound function size: ', Chris_plot_bound_fun.shape)
            print('-------------------------------------------------------------------')
            
            #---------------------------------------------------------------------------#
            # save data 
            run_test_data = {}

            run_test_data['model_plot_data']        = model_plot_data 

            run_test_data['Chris_plot_col_fun']     = Chris_plot_col_fun
            run_test_data['Chris_plot_init_fun']    = Chris_plot_init_fun
            run_test_data['Chris_plot_bound_fun']   = Chris_plot_bound_fun
        
            run_test_data['Prob_plot_col_dist']     = Prob_plot_col_dist
            run_test_data['Prob_plot_init_dist']    = Prob_plot_init_dist
            run_test_data['Prob_plot_bound_dist']   = Prob_plot_bound_dist
            run_test_data['result_folder']          = result_folder
            run_test_data['run_test_data_filename'] = 'run_test_data.mat'
            
            # save the resulting mat file with scipy.io
            sio.savemat(run_test_data['result_folder'] + '/' + run_test_data['run_test_data_filename'], run_test_data) 
            print('save to:',run_test_data['result_folder'] + '/' + run_test_data['run_test_data_filename'])

               
            print('\nComputation time total: {} minutes'.format((time.time()-t1)/60))

