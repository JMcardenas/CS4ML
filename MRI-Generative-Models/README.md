
# Generative Compressed Sensing Experiments

This repository contains code for conducting generative compressed sensing experiments using the braingen-0.1.0 pre-trained progressive generative neural network model from [https://github.com/neuronets/trained-models](https://github.com/neuronets/trained-models). The model is capable of generating T1-weighted brain MR scans.

## Setup

To set up the repository, please follow the instructions below:

1. Clone this repository to your local machine.
2. Open the terminal and navigate to the repository's root directory.
3. Before running the `install.sh` script, make sure to edit line 9 of the script to replace `"your.email@server.com"` with your GitHub email and `"your_github_username"` with your GitHub username. This will configure your Git user details.
   ```shell
   git config --global user.email "your.email@server.com" && git config --global user.name "your_github_username"
   ```
4. After saving the changes to `install.sh`, run the script to download the `braingen-0.1.0` pre-trained model. This will download the model with the image size 128 used for these experiments to the following directory: `trained-models/neuronets/braingen/0.1.0/generator_res_128`.
5. Run the `install.sh` shell script using the following command:
   ```shell
   sudo bash install.sh
   ```

## Running the Code

Once the setup is complete, you can run the code by executing the `run_train_test_local.sh` script. This script will automatically run the `generative_cs_example.py` file with the necessary command-line arguments to generate 20 trials of the experiment at a range of sampling percentages between 0.125% and 2%. The output of the code are individual MATLAB mat files for each trial of the experiment. 

To customize the experiments, you can modify the parameters in the `run_train_test_local.sh` script. The following parameters can be adjusted:

- **Number of trials:** Control the number of experimental trials by modifying "min_trials" and "max_trials".
- **Number of iterations for computing K tilde:** The number of iterations to run of Algorithm 1 from Appendix C to compute K tilde for the optimal sampling density.
- **Number of epochs:** Set the number of epochs for solving the generative compressed sensing least squares problem.
- **Examples to run:** Specify the examples to run within the `generative_cs_example.py` file. Currently only braingen is implemented.

The core code responsible for setting up the experiment, sampling, loading the `braingen-0.1.0` model, and computing the DFT operators is located in the `generative_cs_example.py` file. Feel free to modify this file to suit your specific requirements.

## Generating Figures using MATLAB Plotting Scripts

In addition to the code for running generative compressed sensing experiments, this repository provides MATLAB plotting scripts for generating figures. These scripts are helpful for visualizing the experimental results. Here's how you can utilize them:

### Prerequisites

Before using the MATLAB plotting scripts, make sure you have MATLAB installed on your system.

### Mean PSNR and Shaded Plots

The scripts `plot_book_style.m`, `get_fig_param.m`,  from [https://github.com/simone-brugiapaglia/sparse-hd-book/tree/main/utils/graphics](https://github.com/simone-brugiapaglia/sparse-hd-book/tree/main/utils/graphics), slightly modified for the purpose of plotting these experiments, is used to generate mean PSNR (Peak Signal-to-Noise Ratio) plots along with shaded plots representing one standard deviation.

By default, running the `generative_cs_example.py` code will write the output data in the directory `~/scratch/braingen_GCS_example/run_ID_image_data_128`. Here, `run_ID` is a parameter specified in `run_train_test_local.sh`, which determines the root directory name for a series of runs.

To generate the plots, follow these steps:

1. Open the `plot_book_style.m` MATLAB script.

2. Modify the variable `jmin` on line 12 of the script. Setting this number to 1 generates the left plot for Figure 8, while setting `jmin = 3` generates the left plot of Figure 2.

3. Run the `plot_book_style.m` script to generate the mean PSNR and shaded plots.

### Shaded Plots

To generate the middle plots of Figures 2 and 8, use the script `make_shaded_plot.m`.

Follow these steps to generate the shaded plots:

1. Run the `make_shaded_plot.m` MATLAB script.

2. The script will generate the required plots based on the data obtained from running the `generative_cs_example.py` code.

### 2D K tilde Plot

To generate the 2D plot of K tilde for Figure 2, use the script `plot_2d_samp_patterns.m` after loading the corresponding `K_tilde_lines.mat` file for the experiment.

To generate the 2D plot, follow these steps:

1. Load the `K_tilde_lines.mat` file for the experiment.

2. Run the `plot_2d_samp_patterns.m` MATLAB script.

3. The script will generate the 2D plot of K tilde.

### 3D K tilde Volume Plot

To generate the 3D volume plot of K tilde for Figure 8, use the script `plot_samp_patterns.m` after loading the corresponding `K_tilde.mat` file for the experiment.

To generate the 3D volume plot, follow these steps:

1. Load the `K_tilde.mat` file associated with the specific experiment.

2. Run the `plot_samp_patterns.m` MATLAB script.

3. The script will generate the 3D volume plot of K tilde.

## License

This repository is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code and MATLAB plotting scripts as permitted by the license.

## Acknowledgements

The generative compressed sensing experiments rely on the `braingen-0.1.0` pre-trained progressive generative neural network model, which can be found at [https://github.com/neuronets/trained-models](https://github.com/neuronets/trained-models). The braingen-0.1.0 generative model was trained using the nobrainer framework [https://github.com/neuronets/nobrainer](https://github.com/neuronets/nobrainer) available under Apache License, Version 2.0. 

The MATLAB plotting scripts are from [https://github.com/simone-brugiapaglia/sparse-hd-book/tree/main/utils/graphics](https://github.com/simone-brugiapaglia/sparse-hd-book/tree/main/utils/graphics). 

We extend our gratitude to the authors for providing these valuable resources.

If you have any questions or encounter any issues, please don't hesitate to reach out. Happy experimenting!

