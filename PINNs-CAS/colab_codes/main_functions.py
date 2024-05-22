import tensorflow as tf
import numpy as np 
import random 
import os, io, math
import scipy.io as sio

def init_model(nb_hidden_layers, nb_nodes_per_layer, input_dim, output_dim, activation, initializer, trial):
    # Initialize a feedforward neural network
    model = tf.keras.Sequential()

    # Input is two-dimensional (time + one spatial dimension)
    input_layer = tf.keras.layers.InputLayer(input_shape= (input_dim,), name='input_layer')
    model.add(input_layer)

    #------------------------------------------------------------------------------#
    # set up the initializer for the weight and biase
    #------------------------------------------------------------------------------#
    sigma = 1e-1
    #trial = 1

    if initializer == 'normal':
        weight_bias_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev= sigma, seed= trial)
    elif initializer == 'uniform':
        weight_bias_initializer = tf.keras.initializers.RandomUniform(minval=-sigma, maxval= sigma, seed= trial)
    elif initializer == 'constant':
        weight_bias_initializer = tf.keras.initializers.Constant(value= sigma)
    elif initializer == 'he_normal':
        weight_bias_initializer = tf.keras.initializers.HeNormal(seed= trial)        
    elif initializer == 'xavier_normal':
        weight_bias_initializer = tf.keras.initializers.GlorotNormal(seed= trial) 
    elif initializer == 'xavier_uniform':
        weight_bias_initializer = tf.keras.initializers.GlorotUniform(seed= trial)     
    else: 
        print('initializer must be one of the supported types, e.g. normal, uniform, constant, xavier, etc.')

    # Append hidden layers
    all_layers = []

    for layer in range(nb_hidden_layers):
        if layer == nb_hidden_layers-1:
            layer_name = 'first_to_last_hidden_layer'
        else:
            layer_name = 'hidden_layer_' + str(layer)

        hidden_layer = tf.keras.layers.Dense(nb_nodes_per_layer, 
                                            activation= activation,
                                            name= layer_name,
                                            kernel_initializer= weight_bias_initializer,
                                            bias_initializer= weight_bias_initializer)
                                            #dtype=tf.float32)
        all_layers.append(hidden_layer)
        model.add(hidden_layer)

    # Output dim
    output_layer = tf.keras.layers.Dense(output_dim,
                                         activation= tf.keras.activations.linear,
                                         trainable= True,
                                         use_bias= False,
                                         name= 'output_layer',
                                         kernel_initializer= weight_bias_initializer
                                        )
    model.add(output_layer)
    
    return model

# Extract from the input layer until the last hidden layer of the model 
def extract_model(model):
    # To do this you have to save your model with first to last hidden layer 
    B_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('first_to_last_hidden_layer').output)

    return B_model

def get_M_and_k_values(nb_nodes_per_layer, input_dim, nb_train_pts, nb_training_steps, N_max, type_data):
    # nb_nodes_per_layer................ number of nodes per layer
    # input_dim......................... input dimension of DNN
    # nb_train_pts...................... number of training points
    # nb_training_steps................. number of training steps 
    # N_max............................. numerical dimension ('r')

    #1. Check the grid/nb_train_pts size
    if input_dim < 4: 
            if 1000 < nb_train_pts: 
                print('Input dim: ',str(input_dim), 'Grid size: ', str(nb_train_pts), 'correct size for dim')
            else: 
                print('increase grid size')
    elif 4 <= input_dim < 8:
            if 2000 < nb_train_pts:
                print('Input dim: ',str(input_dim), 'Grid size: ', str(nb_train_pts), 'correct size for dim')
            else: 
                print('increase grid size')
    elif 8 <= input_dim:
            if 5000 < nb_train_pts:
                print('Input dim: ',str(input_dim), 'Grid size: ', str(nb_train_pts), 'correct size for dim')
            else: 
                print('increase grid size') 

    #2. Compute sampling values (M_i = k_i N_max, N_max = numerical dimension)

    if input_dim <4:
        if type_data == 'collocation':
            min_M_value = 400
        elif type_data == 'initial':
            min_M_value = 100
        elif type_data == 'boundary':
            min_M_value = 100;
        else: 
            print('wrong type of data')
    elif 4<= input_dim < 8:
        if type_data == 'collocation':
            min_M_value = 600
        elif type_data == 'initial':
            min_M_value = 300
        elif type_data == 'boundary':
            min_M_value = 300;
        else: 
            print('wrong type of data')
    elif 8 <= input_dim:
        if type_data == 'collocation':
            min_M_value = 1000
        elif type_data == 'initial':
            min_M_value = 700
        elif type_data == 'boundary':
            min_M_value = 700;
        else: 
            print('wrong type of data')

    k_init  = np.round(min_M_value/N_max)
    k_final = np.round(nb_train_pts/(2*N_max))

    #print('k init value: ', str(k_init), 'k final value', str(k_final))
    k_values = np.round(np.linspace(k_init,k_final,nb_training_steps-1).astype(int))
    M_values = (k_values*N_max).astype(int)

    return M_values, k_values    

def evaluate_data_model(model, data):
    # model............. DNN model
    # data.............. training/testing data

    t = data[:,0:1]
    x = data[:,1:2]
    
    return  model(tf.stack([t[:,0], x[:,0]], axis=1))    

def get_measurement_matrix(model, data, nb_train_pts):
    # model............. DNN model
    # data.............. training/testing data
    B_model            = extract_model(model)
    measurement_matrix = evaluate_data_model(B_model, data)/np.sqrt(nb_train_pts)

    return measurement_matrix    

def get_numerical_dim(B_matrix, nb_nodes):
    # B_matrix.............measurement matrix of data on B model
    # nb_nodes.............number of nodes

    # create the measurement matrix 
    #B_matrix = evaluate_data_model(model, data)/tf.sqrt(nb_pts)

    # compute svd
    S, U, V = tf.linalg.svd(B_matrix, full_matrices=False, compute_uv=True, name=None)

    # find rank/numerical dimension
    for i in range(nb_nodes):
        if (S[i]/S[0]) > 1e-6:
            r = i
    
    # shift 
    r = r + 1

    return  U, r    

def get_k_ratios(M_values, k_values, nb_nodes, r, iter):
    # M_values........sampling numbers
    # k_values........sampling ratios
    # nb_nodes........number of nodes
    # r...............numerical dimension
    # iter............number of current iteration

    if nb_nodes == r:
        #print('rank is equal to N')
        N_max       = r
        k_min_ratio = 0

        if iter == 0:
            k_max_ratio = (k_values[iter]).astype(int)
            s_l         = (M_values[iter] - k_max_ratio*r).astype(int) 
        else:
            k_max_ratio = (k_values[iter]-k_values[iter-1]).astype(int)
            s_l         = (M_values[iter] - M_values[iter-1]-k_max_ratio*r).astype(int)

    else:
        #print('rank is less to N: ', r)
        N_max       = r
        k_min_ratio = 1

        if iter == 0:
            k_max_ratio = np.floor(M_values[iter]/r).astype(int)
            s_l         = (M_values[iter] - k_max_ratio*r).astype(int)
        else:
            k_max_ratio = np.floor((M_values[iter]-M_values[iter-1])/r).astype(int)
            s_l         = (M_values[iter] - M_values[iter-1]-k_max_ratio*r).astype(int)

    #print('k_max_ratio: ', k_max_ratio, 'k_min_ratio: ', k_min_ratio)

    return k_max_ratio, k_min_ratio, s_l    

def CAS_method(U_matrix, r, s_l, k_min_ratio, k_max_ratio, nb_pts, nb_nodes, iter, I_grid, I_new, I_ad, I, weights_opt):
    # r .............numerical dimension
    # U_matrix.......U matrix from svd of B_matrix
    # nb_pts.........number of points 
    # iter............number of current iteration    

    # compute probability distribution
    mu = abs(np.square(U_matrix[:,0:r]))
    #print('size mu ' + str(mu.shape) + ' | nb of pts: ' + str(nb_pts))

    if iter == 0:
        # Draw pts from current prob. dist. 
        for j in range(r):
            mu_j  = mu[:,j]
            I_a   = random.choices(I_grid, weights = mu_j, k = k_max_ratio)
            I_new = np.append(I_new,I_a)

        #print('size I new: ' + str((I_new).astype(int).shape[0]))

        if r < nb_nodes:
            #print('Draw k_min samples from mu_j')
            count_pts = 0
            samp_num  = 0
            while count_pts < s_l:
                mu_j  = mu[:,samp_num]
                I_a   = random.choices(I_grid, weights = mu_j, k = k_min_ratio)
                I_new = np.append(I_new, I_a)

                count_pts = count_pts + 1
                
                if samp_num == (r-1):
                    samp_num = 0
                else:
                    samp_num = samp_num + 1

        #print('size I new: ' + str((I_new).astype(int).shape[0]))
    else: 
        # Add points from old sampling measures
        I_ad = np.array([])

        #print('k_ad: '+ str(k_max_ratio))

        for j in range(r):
            mu_j    = mu[:,j]
            I_ad_ax = random.choices(I_grid, weights = mu_j, k = k_max_ratio)
            I_ad    = np.append(I_ad, I_ad_ax)

        if r < nb_nodes:
            #print('Draw k_min samples from mu_i')
            count_pts = 0
            samp_num  = 0
            while count_pts < s_l:
                mu_j     = mu[:,samp_num]
                I_ad_ax  = random.choices(I_grid, weights = mu_j, k = k_min_ratio)
                I_ad     = np.append(I_ad, I_ad_ax)

                count_pts = count_pts + 1
                if samp_num == (r-1):
                    samp_num = 0
                else:
                    samp_num = samp_num + 1

        I_ad = np.array(I_ad, dtype= np.float32).astype(int)
        #print('size I ad: ' + str(I_ad.shape[0]))

    if iter == 0:
        I = I_new
    else:
        I = np.append(I, I_ad)
    
    I = I.astype(int)

    # Compute weights
    if weights_opt:
        Chris_func = np.sum(mu[I,:], axis=1)/r
        weights    = np.sqrt(np.divide(1,Chris_func))
    else:
        weights = None

    return I, weights   # return I

def MC_method(M_values, iter, I):
    # M_values................. sampling numbers
    # iter..................... current iteration number

    if iter == 0: 
        I = np.arange(0,M_values[iter])
    else: 
        I_ad = np.arange(M_values[iter-1], M_values[iter])
        I    = np.append(I, I_ad)

    # convert I 
    I = np.array(I, dtype=np.float32).astype(int)
    
    # Compute weights
    weights = None

    return I, weights    

