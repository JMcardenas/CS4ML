import numpy as np
import os, argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import random as python_random
from pathlib import Path
import nibabel as nib
from nilearn import plotting
import hdf5storage
import calendar
import datetime
import pywt
from IPython import display
import scipy.io as sio
import time


"""
Translation of the partialFourier3D function from MATLAB to NumPy (here idx1-3 need to be indexed from 0 (1 less than MATLAB)).
"""
def partialFourier3D(inds,N,x,mode):

    if mode == 1:
        x = np.reshape(x,(N,N,N))
        z = np.fft.fftn(x) / N**(3/2)
        z = np.fft.fftshift(z)
        y = z.flat[inds]
    else:
        z = np.zeros((N,N,N))
        z = z.astype(complex)
        z.flat[inds] = x
        z = np.fft.ifftshift(z)
        y = np.fft.ifftn(z) * N**(3/2)
        y = np.reshape(y,(N*N*N,1))

    return y

"""
Translation of the partialFourier3D function from MATLAB to TensorFlow (here idx1-3 need to be indexed from 0 (1 less than MATLAB)).
"""
def partialFourier3DTF(inds,N,x,mode):

    if mode == 1:
        x = tf.reshape(x,[N,N,N])
        z = tf.signal.fft3d(tf.cast(x, dtype=tf.complex64)) / N**(3/2)
        z = tf.signal.fftshift(z)
        w = tf.reshape(z,[N*N*N,1])
        y = tf.gather(w,inds_flattened)
    else:
        z = tf.zeros([N,N,N])
        z = z.astype(complex)
        w = tf.reshape(z,[N*N*N,1])
        ww = tf.gather(w,inds_flattened)
        ww = x
        z = ww
        z = tf.signal.ifftshift(z)
        y = tf.signal.ifftn(z) * N**(3/2)
        y = tf.reshape(y,[N*N*N,1])

    return y

"""
Code for generating the sampling paterns
"""
def SampMatrix3D(N, m, mode, DNN_run_data):

    if mode == 1: # optimal sampling in k-space
        # Initialize K tilde data dict
        K_tilde_data = {}

        # Initialize K tilde
        K_tilde = np.zeros((N**3,))

        max_samples_K_tilde = DNN_run_data["max_samples_K_tilde"]

        for i in range(max_samples_K_tilde):
            
            # store previous
            K_tilde_prev = K_tilde

            # generate elements from the range of the generative model
            latent_1 = tf.random.normal((1, 1024))
            img1 = generate(latent_1)["generated"]
            img_squeezed1 = np.squeeze(img1)
            latent_2 = tf.random.normal((1, 1024))
            img2 = generate(latent_2)["generated"]
            img_squeezed2 = np.squeeze(img2)

            # Compute the difference between two elements in the range of the generative model
            x_diff = img_squeezed1 - img_squeezed2

            # Compute the norm
            x_diff_norm = np.linalg.norm(np.ndarray.flatten(x_diff),2)

            # Compute the sampling densities
            all_inds = np.arange(0, N**3)
            F_x_diff = partialFourier3D(all_inds,N,x_diff,1)*np.sqrt(N**3)

            a_vals = np.square(np.absolute(F_x_diff))/x_diff_norm**2
            # update K_tilde
            K_tilde = np.maximum(K_tilde_prev, a_vals)
            # record update diff
            K_tilde_iter_2_norm_diff = np.linalg.norm(K_tilde - K_tilde_prev,2)

            print(i, K_tilde, K_tilde_iter_2_norm_diff)
            #CMCS = np.amax(K_tilde)
            #CCS = np.sum(K_tilde)/N**3
            #print('sanity check:', CMCS, CCS)

            # store the K_tilde data
            if i == 0:
                K_tilde_data['K_tilde_iterations'] = K_tilde
                K_tilde_data['K_tilde_iter_2_norm_diffs'] = K_tilde_iter_2_norm_diff
            else:
                K_tilde_data['K_tilde_iter_2_norm_diffs'] = np.vstack([
                            K_tilde_data['K_tilde_iter_2_norm_diffs'], K_tilde_iter_2_norm_diff])
                if i % 10 == 0 or i == (max_samples_K_tilde - 1):
                    K_tilde_data['K_tilde_iterations'] = np.vstack([
                            K_tilde_data['K_tilde_iterations'], K_tilde])

        # compute the probability distribution from K_tilde
        prob = K_tilde/np.sum(K_tilde)
        # sample k space randomly according to this distribution
        inds = np.random.choice(N**3, m, replace=False, p=prob) 

        # form the 3d sampling mask matrix as N^3 vector
        R = np.zeros((N**3,1))
        R[inds] = 1
        # reshape to 3d
        R = np.reshape(R,(N,N,N))

        # add the zero frequency
        mid = int(N/2)
        R[mid,mid,mid] = 1 

        # record the indices of the sampling mask
        inds = np.argwhere(R==1)

        # save data for multiple runs
        K_tilde_data['K_tilde'] = K_tilde
        K_tilde_data['inds'] = inds
        K_tilde_data['R'] = R
        K_tilde_data['prob'] = prob
        sio.savemat(DNN_run_data['pathname'] + '/K_tilde.mat', K_tilde_data)

        # update the sampling probability distribution for computing weights for LS problem
        DNN_run_data['prob'] = prob

    if mode == 2: # uniform random sampling in k-space

        # generate uniform random 3d sampling mask
        R = np.zeros((N**3,1))
        r = np.random.permutation(N**3)
        r = r[0:m]
        R[r] = 1
        R = np.reshape(R,(N,N,N))

        # add the zero frequency
        mid = int(N/2)
        R[mid,mid,mid] = 1 # Include zero frequency (changed here from N/2+1 to N/2 from MATLAB

        # record the indices of the sampling mask
        inds = np.argwhere(R==1)

        # compute the uniform sampling density for k-space
        K_tilde = np.ones((N**3,))
        prob = K_tilde/np.sum(K_tilde)

        # update the sampling probability distribution for computing weights for LS problem
        DNN_run_data['prob'] = prob

    if mode == 3: # line sampling in k-space

        # Initialize K tilde data dict
        K_tilde_data = {}

        # Initialize K tilde
        K_tilde_2d = np.zeros((N**2,))

        max_samples_K_tilde = DNN_run_data["max_samples_K_tilde"]

        for i in range(max_samples_K_tilde):
            # store previous
            K_tilde_2d_prev = K_tilde_2d

            # generate elements from the range of the generative model
            latent_1 = tf.random.normal((1, 1024))
            img1 = generate(latent_1)["generated"]
            img_squeezed1 = np.squeeze(img1)
            latent_2 = tf.random.normal((1, 1024))
            img2 = generate(latent_2)["generated"]
            img_squeezed2 = np.squeeze(img2)

            # Compute the difference between two elements in the range of the generative model
            x_diff = img_squeezed1 - img_squeezed2

            # Compute the norm
            x_diff_norm = np.linalg.norm(np.ndarray.flatten(x_diff),2)

            # Compute the sampling densities
            all_inds = np.arange(0, N**3)
            F_x_diff = partialFourier3D(all_inds,N,x_diff,1)*N

            F_x_diff_reshaped = np.reshape(F_x_diff,(N,N,N))
            F_x_diff_res = np.square(np.absolute(F_x_diff_reshaped))
            a_vals = np.zeros((N,N))
            for j in range(N):
                for k in range(N):
                    a_vals[j,k] = np.sum(F_x_diff_res[:,j,k], axis = 0)
            # compute corresponding a values
            a_vals = a_vals/x_diff_norm**2
            # reshape to N^2 vector
            a_vals = np.reshape(a_vals,(N**2,))
            # update K_tilde_2d
            K_tilde_2d = np.maximum(K_tilde_2d_prev, a_vals)
            # record update diff
            K_tilde_iter_2_norm_diff = np.linalg.norm(K_tilde_2d - K_tilde_2d_prev,2)

            print(i, K_tilde_2d, K_tilde_iter_2_norm_diff)
            #CMCS = np.amax(K_tilde_2d)
            #CCS = np.sum(K_tilde_2d)/N**2
            #print('sanity check:', CMCS, CCS)

            # store the K_tilde data
            if i == 0:
                K_tilde_data['K_tilde_iterations'] = K_tilde_2d
                K_tilde_data['K_tilde_iter_2_norm_diffs'] = K_tilde_iter_2_norm_diff
            else:
                K_tilde_data['K_tilde_iter_2_norm_diffs'] = np.vstack([
                            K_tilde_data['K_tilde_iter_2_norm_diffs'], K_tilde_iter_2_norm_diff])
                if i % 10 == 0 or i == (max_samples_K_tilde - 1):
                    K_tilde_data['K_tilde_iterations'] = np.vstack([
                            K_tilde_data['K_tilde_iterations'], K_tilde_2d])

        # compute the 2d probability distribution for line k-space sampling with K_tilde
        prob_2d = K_tilde_2d/np.sum(K_tilde_2d)
        # sample the 2d plane from this distribution for the line sampling
        inds_2d = np.random.choice(N**2, m, replace=False, p=prob_2d) 

        # form the 2d sampling mask to be repeated along first direction
        R_2d = np.zeros((N**2,1))
        R_2d[inds_2d] = 1
        R_2d = np.reshape(R_2d,(N,N))

        # record the 2d coords for updating the 3d matrix
        inds_2d_coords = np.argwhere(R_2d==1)

        # form the 3d sampling mask matrix
        R = np.zeros((N,N,N))
        # fully sample in first dimension
        for l in range(m):
            R[:,inds_2d_coords[l,0], inds_2d_coords[l,1]] = 1

        # add the zero frequency
        mid = int(N/2)
        R[mid,mid,mid] = 1 

        # record the indices of the sampling mask
        inds = np.argwhere(R==1)

        # form 3d K_tilde for weighted LS problem
        #K_tilde_2d_reshaped = np.reshape(K_tilde_2d,(N,N))
        prob_2d_reshaped = np.reshape(prob_2d,(N,N))
        prob = np.zeros((N,N,N))
        for i in range(N):
            prob[i,:,:] = prob_2d_reshaped

        # save data for multiple runs
        K_tilde_data['K_tilde_2d'] = K_tilde_2d
        K_tilde_data['inds'] = inds
        K_tilde_data['R'] = R
        K_tilde_data['prob'] = prob
        K_tilde_data['prob_2d'] = prob_2d
        sio.savemat(DNN_run_data['pathname'] + '/K_tilde_lines.mat', K_tilde_data)

        # update the sampling probability distribution for computing weights for LS problem
        DNN_run_data['prob'] = prob

    if mode == 4: # uniform random line sampling in k-space
        R_2d = np.zeros((N**2,1))
        r = np.random.permutation(N**2)
        r = r[0:m]
        R_2d[r] = 1
        R_2d = np.reshape(R_2d,(N,N))
        inds_2d_coords = np.argwhere(R_2d==1)
        R = np.zeros((N,N,N))
        for l in range(m):
            R[:,inds_2d_coords[l,0],inds_2d_coords[l,1]] = 1
        mid = int(N/2)
        R[mid,mid,mid] = 1 # Include zero frequency (changed here from N/2+1 to N/2 from MATLAB
        # Construct corresponding indices
        inds = np.argwhere(R==1)
        K_tilde = np.ones((N**3,))
        prob = K_tilde/np.sum(K_tilde)
        DNN_run_data['prob'] = prob

    return inds, R

"""
Utility for implementing an equivalent function to MATLAB ind2sub
"""
def ind2sub(sz, ind):
    np.ravel_multi_index(ind, dims=sz, order='F')

"""
Utility to compute the PSNR of a 2d image
"""
def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    frame_mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if frame_mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(frame_mse)))

"""
Custom tensor Adam optimizer
"""
class TensorAdamOptimizer:
    def __init__(self, stepsize=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-10):
        self.stepsize = stepsize
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.time = 0
        self.first_moment = None
        self.second_moment = None

    def init(self, shape):
        self.first_moment = tf.zeros(shape)
        self.second_moment = tf.zeros(shape)

    def calculate_update(self, gradient):
        if self.first_moment is None or self.second_moment is None:
            self.init(tf.shape(gradient))
        self.time = self.time + 1
        self.first_moment = self.beta_1 * self.first_moment + (1 - self.beta_1) * gradient
        self.second_moment = self.beta_2 * self.second_moment + (1 - self.beta_1) * (gradient**2)
        first_moment_corrected = self.first_moment / (1 - self.beta_1**self.time)
        second_moment_corrected = self.second_moment / (1 - self.beta_2**self.time)
        return self.stepsize * first_moment_corrected / (tf.sqrt(second_moment_corrected) + self.eps)

    def reset(self):
        self.first_moment = tf.zeros_like(self.first_moment)
        self.second_moment = tf.zeros_like(self.second_moment)

"""
Gradient descent for solving the generative CS problem
"""
def gradient_descent(opt, y, A, G, init, x_true, DNN_run_data):
    z_tensor = tf.convert_to_tensor(init, dtype = float, name ='the_z_tensor')
    iter = 1
    gradf = 100.0
    lrn_rate = 1.0e2
    y_weighted = tf.math.multiply(DNN_run_data['LS_weights'], y)

    while gradf >= DNN_run_data['error_tol'] and iter <= DNN_run_data['nb_epochs']:

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(z_tensor)
            AGz_reshaped = tf.reshape(A(G(z_tensor)), [DNN_run_data['m_samples'],])
            AGz_reshaped_weighted = tf.math.multiply(DNN_run_data['LS_weights'], AGz_reshaped)
            loss = tf.norm(AGz_reshaped_weighted - y_weighted,2)**2#/len(y)

        grads = tape.gradient(loss,z_tensor)
        z_tensor = z_tensor - lrn_rate * opt.calculate_update(grads)
        gradf = np.linalg.norm(grads.numpy(), 2)
        l2_norm_latent_err = np.linalg.norm(z_tensor - x_true)
        print('iter', iter, 'loss', np.abs(loss.numpy()), 'gradf', gradf, '2-norm iter err', l2_norm_latent_err)

        if iter == 1:
            DNN_run_data['loss'] = np.abs(loss.numpy())
            DNN_run_data['iters'] = iter
            DNN_run_data['gradf'] = gradf
            DNN_run_data['l2_norm_latent_errs'] = l2_norm_latent_err
        else:
            DNN_run_data['loss'] = np.vstack([DNN_run_data['loss'], np.abs(loss.numpy())])
            DNN_run_data['iters'] = np.vstack([DNN_run_data['iters'], iter])
            DNN_run_data['gradf'] = np.vstack([DNN_run_data['gradf'], gradf])
            DNN_run_data['l2_norm_latent_errs'] = np.vstack([DNN_run_data['l2_norm_latent_errs'], 
                                                                l2_norm_latent_err])

        if DNN_run_data['plot_images']: 
            x_plot = G(z_tensor)
            img_recov = np.reshape(x_plot.numpy().real,(N,N,N))
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            img_plot = nib.Nifti1Image(img_recov.astype(np.uint8), np.eye(4))
            display.clear_output(wait=True)
            plotting.plot_anat(anat_img=img_plot, cut_coords=(N/2,N/2,N/2), figure=fig, axes=ax, 
                                draw_cross=False,
                                title=model_path.name.split("_")[-1])
            display.display(plt.gcf())
            plt.savefig(pathname + '/iter_' + str(iter).rjust(10,'0') + '.png')

        iter += 1

    return z_tensor

"""
Main method
"""
if __name__ == '__main__': 

    # parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_epochs", default = 10000, type = int, help = "Number of epochs for training")
    parser.add_argument("--run_ID", type = str, help = "String for naming batch of trials in this run")
    parser.add_argument("--example", type = str, help = "Example function to approximate")
    parser.add_argument("--optimizer", default = 'Adam', type = str, help = "Optimizer to use in minimizing the loss")
    parser.add_argument("--precision", default = 'single', type = str, help = "Precision for variables")
    parser.add_argument("--quiet", default = 0, type = int, help = "Switch for verbose output")
    parser.add_argument("--trial_num", default = 0, type = int, help = "Number for the trial to run")
    parser.add_argument("--weighted_LS", default = 0, type = int, help = "Switch for solving the weighted LS problem: 0 = unweighted, 1 = weighted")
    parser.add_argument("--samp_perc", default = 0.05, type = float, help = "Sampling percentage")
    parser.add_argument("--samp_method", default = 1, type = int, help = "Sampling method: 1 = optimal, 2 = uniform")
    parser.add_argument("--max_samples_K_tilde", default = 1000, type = int, help = "Number of samples to use in computing K tilde")
    parser.add_argument("--error_tol", default = 5e-7, type = float, help = "Stopping tolerance for the solvers")
    parser.add_argument("--lrn_rate_schedule", default = "exp_decay", type = str, help = "Learning rate schedule, e.g. exp_decay")
    args = parser.parse_args()

    if args.run_ID is None:
        date = datetime.datetime.utcnow()
        utc_time = calendar.timegm(date.utctimetuple())
        key = 'run_' + str(utc_time)
    else:
        key = args.run_ID

    print('running with key:', key)

    print('samp_perc:', args.samp_perc)
    print('samp_method:', args.samp_method)

    # record the trial number
    trial = args.trial_num
    print('running trial:', trial)

    N = 128
    pathname = '/home/nickdexter/scratch/braingen_GCS_example/' + key + '_image_data_' + str(N)
    if not os.path.exists(pathname):
       os.makedirs(pathname)

    print('writing to:', pathname)

    model_pathname = "/home/nickdexter/Dropbox/Generative_CS/code/trained-models/neuronets/braingen/0.1.0/generator_res_" + str(N)
    model_path = Path(model_pathname)

    print('loading model from:', model_pathname)

    generator = tf.keras.models.load_model(str(model_path)+str('/weights'))
    #generator = tf.saved_model.load(str(model_path)+str('/weights'))
    generate = generator.signatures["serving_default"]

    print(generator)
    generator.summary()

    tf.random.set_seed(12345)
    latent = tf.random.normal((1, 1024))
    img = generate(latent)["generated"]
    img_squeezed = np.squeeze(img)
    orig_latent = latent.numpy()

    start_time = time.time()
    tf.random.set_seed(trial)
    np.random.seed(trial)
    python_random.seed(trial)

    DNN_run_data = {}
    DNN_run_data['key'] = key
    DNN_run_data['start_time'] = start_time
    DNN_run_data['pathname'] = pathname
    DNN_run_data['model_pathname'] = model_pathname
    DNN_run_data['orig_latent'] = orig_latent
    DNN_run_data['decay_steps'] = 1e3
    DNN_run_data['nb_epochs']   = args.nb_epochs
    DNN_run_data['error_tol']   = args.error_tol
    DNN_run_data['init_rate']   = 1e-3
    DNN_run_data['plot_images'] = False
    DNN_run_data['samp_method'] = args.samp_method
    DNN_run_data['samp_perc'] = args.samp_perc
    DNN_run_data['max_samples_K_tilde'] = args.max_samples_K_tilde
    DNN_run_data['N'] = N


    if DNN_run_data['samp_method'] == 1: # optimal sampling
        m_samples = np.ceil(N**3*DNN_run_data['samp_perc']).astype(int)
        print('using', str(m_samples), ' GCS samples for reconstruction')
        if not os.path.isfile(pathname + '/K_tilde.mat'):
            inds, R = SampMatrix3D(N, m_samples, DNN_run_data['samp_method'], DNN_run_data)
        else:
            K_tilde_data = hdf5storage.loadmat(pathname + '/K_tilde.mat')
            K_tilde = K_tilde_data['K_tilde']
            prob = K_tilde_data['prob']
            DNN_run_data['prob'] = prob
            DNN_run_data['K_tilde'] = K_tilde
            print('K_tilde shape =', K_tilde.shape)
            inds = np.random.choice(N**3, m_samples, replace=False, p=np.reshape(prob,(N**3,))) 
            R = np.zeros((N**3,1))
            R[inds] = 1
            R = np.reshape(R,(N,N,N))

            # add the zero frequency
            mid = int(N/2)
            R[mid,mid,mid] = 1 # Include zero frequency (changed here from N/2+1 to N/2 from MATLAB
            # record the indices of the sampling mask
            inds = np.argwhere(R==1)

    elif DNN_run_data['samp_method'] == 2: # uniform random sampling
        m_samples = np.ceil(N**3*DNN_run_data['samp_perc']).astype(int)
        print('using', str(m_samples), 'uniform random samples for reconstruction')
        inds, R = SampMatrix3D(N, m_samples, DNN_run_data['samp_method'], DNN_run_data)

    elif DNN_run_data['samp_method'] == 3: # optimal sampling along lines in k-space
        m_samples_2d = np.ceil(N**2*DNN_run_data['samp_perc']).astype(int)
        m_samples = m_samples_2d*N
        print('using', str(m_samples), 'k-space line GCS samples for reconstruction, with samp_perc', DNN_run_data['samp_perc'], 'and', m_samples_2d, '2d samples')
        if not os.path.isfile(pathname + '/K_tilde_lines.mat'):
            inds, R = SampMatrix3D(N, m_samples_2d, DNN_run_data['samp_method'], DNN_run_data)
        else:
            # load the data
            K_tilde_data = hdf5storage.loadmat(pathname + '/K_tilde_lines.mat')
            K_tilde_2d = K_tilde_data['K_tilde_2d']
            prob = K_tilde_data['prob']
            prob_2d = K_tilde_data['prob_2d']
            DNN_run_data['prob'] = prob

            # generate random 2d indices according to K_tilde_2d
            inds_2d = np.random.choice(N**2, m_samples_2d, replace=False, p=np.reshape(prob_2d,(N**2,))) 

            R_2d = np.zeros((N**2,1))
            R_2d[inds_2d] = 1
            R_2d = np.reshape(R_2d,(N,N))
            inds_2d_coords = np.argwhere(R_2d==1)
            R = np.zeros((N,N,N))
            for l in range(m_samples_2d):
                R[:,inds_2d_coords[l,0],inds_2d_coords[l,1]] = 1

            # add the zero frequency
            mid = int(N/2)
            R[mid,mid,mid] = 1 # Include zero frequency (changed here from N/2+1 to N/2 from MATLAB
            # record the indices of the sampling mask
            inds = np.argwhere(R==1)

    elif DNN_run_data['samp_method'] == 4: # unform random line sampling in k-space
        m_samples_2d = np.ceil(N**2*DNN_run_data['samp_perc']).astype(int)
        m_samples = m_samples_2d*N
        print('using', str(m_samples), ' k-space line uniform random samples for reconstruction, with samp_perc', DNN_run_data['samp_perc'], 'and', m_samples_2d, '2d samples')
        inds, R = SampMatrix3D(N, m_samples_2d, DNN_run_data['samp_method'], DNN_run_data)


    DNN_run_data['m_samples'] = m_samples
    DNN_run_data['inds'] = inds
    DNN_run_data['R'] = R

    inds_flattened = np.zeros((len(inds)))
    for i in range(len(inds)):
        inds_flattened[i] = np.ravel_multi_index(inds[i,:],(N,N,N))
    inds_flattened = inds_flattened.astype(int)
    inds_tensor = tf.convert_to_tensor(inds_flattened, dtype = tf.int32)

    prob = DNN_run_data['prob']
    prob_reshaped = np.reshape(prob,(N**3,))

    # set the weights for the weighted LS problem
    if args.weighted_LS:
        print('running weighted LS problem, weighted_LS =', args.weighted_LS)
        LS_weights = 1.0/np.sqrt(prob_reshaped[inds_flattened])
    else:
        print('not running weighted LS problem, weighted_LS =', args.weighted_LS)
        LS_weights = np.ones(len(inds))

    DNN_run_data['LS_weights'] = LS_weights

    x_data = np.reshape(img_squeezed, (N*N*N))
    x_data = x_data.astype(np.complex64)
    x_data_tensor = tf.convert_to_tensor(x_data, dtype = tf.complex64)

    # reference for generator
    G = lambda x: tf.cast(tf.squeeze(generate(x)["generated"]), dtype = tf.complex64)

    # reference for subsampled Fourier transforms
    A = lambda x: partialFourier3DTF(inds_tensor,N,x,1)*np.sqrt(N**3/m_samples)
    At = lambda x: partialFourier3DTF(inds_tensor,N,x,2)*np.sqrt(m_samples/N**3)

    # generate the measurements
    y = A(x_data_tensor)
    if m_samples != len(y):
        m_samples = len(y)
        DNN_run_data['m_samples'] = m_samples
    y = tf.reshape(y, [m_samples,])
    DNN_run_data['y'] = y.numpy()

    optimizer_to_use = 'gd'
    tau = 0.1
    x_iter = np.zeros((N**3,1))
    u_iter = np.zeros((N**3,1))

    if args.lrn_rate_schedule == "exp_decay":
        # calculate the base so that the learning rate schedule with 
        # exponential decay follows (init_rate)*(base)^(current_epoch/decay_steps)
        DNN_run_data['base'] = np.exp(DNN_run_data['decay_steps']/DNN_run_data['nb_epochs']
                *(np.log(DNN_run_data['error_tol'])-np.log(DNN_run_data['init_rate'])))

        # based on the above, the final learning rate is (init_rate)*(base)^(total_epochs/decay_steps)
        print('based on init_rate = ' + str(DNN_run_data['init_rate'])
            + ', decay_steps = ' + str(DNN_run_data['decay_steps'])
            + ', calculated base = ' + str(DNN_run_data['base']) 
            + ', so that after ' + str(DNN_run_data['nb_epochs'])
            + ' epochs, we have final learning rate = '
            + str(DNN_run_data['init_rate']*DNN_run_data['base']**(DNN_run_data['nb_epochs']/DNN_run_data['decay_steps'])))
        decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            DNN_run_data['init_rate'], DNN_run_data['decay_steps'], DNN_run_data['base'], staircase=False, name=None
        )
    else:
        decay_schedule = 1e-3

    use_tensor_optimizer = True

    if use_tensor_optimizer:
        opt = TensorAdamOptimizer()
    else:
        opt = tf.keras.optimizers.Adam(
            learning_rate=decay_schedule,
            beta_1=0.9, beta_2=0.999, epsilon=1e-07, #amsgrad=False,
            name='Adam')

    if optimizer_to_use == 'fb':
        # forward backward splitting using full image (not what we want)
        for iter in range(3000):
            u_iter = A(x_iter) - y
            #u_iter = partialFourier3D(inds_flattened,N,x_iter,1) - y
            x_iter = x_iter - tau * At(u_iter)
            #x_iter = x_iter - tau * partialFourier3D(inds_flattened,N,u_iter,2)
            x_iter = pywt.threshold(x_iter,tau,'soft')
            if iter % 100 == 0:
                print("iter =", iter)
    else:
        # generate initial z
        z_init = np.random.randn(1, 1024)
        z_iter = gradient_descent(opt, y, A, G, z_init, latent, DNN_run_data)
        x_iter = G(tf.convert_to_tensor(z_iter))

    img_recov = np.reshape(x_iter.numpy().real,(N,N,N))
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    img_plot = nib.Nifti1Image(img_recov.astype(np.uint8), np.eye(4))
    plotting.plot_anat(anat_img=img_plot, cut_coords=(N/2,N/2,N/2), figure=fig, axes=ax, 
                        draw_cross=False,
                        title=model_path.name.split("_")[-1])

    psnr_values = np.zeros((N,))

    for i in range(N):
        X_recov_i = img_recov[:,:,i]
        X_orig_i = img_squeezed[:,:,i]
        psnr_frame_i = calculate_psnr(X_recov_i, X_orig_i, 255)
        psnr_values[i] = psnr_frame_i

    print(psnr_values[30:90])
    DNN_run_data['runtime'] = time.time() - start_time
    DNN_run_data['X_orig'] = img_squeezed
    DNN_run_data['X_recov'] = img_recov
    DNN_run_data['psnr_values'] = psnr_values
    sio.savemat(pathname + '/DNN_run_data_' + str(DNN_run_data['samp_perc']) + 
            '_method_' + str(DNN_run_data['samp_method']) + '_trial_' + str(trial) + '.mat', DNN_run_data)
