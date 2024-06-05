# Hermite Sparse Regression Experiment

This directory contains the MATLAB code and generated figures for the experiments in the paper "A Unified Framework for Learning with Nonlinear Model Classes from Arbitrary Linear Samples" by Ben Adcock, Juan M. Cardenas, and Nick Dexter.

## Files

### MATLAB Code

- **active_learning_exp.m**
  - Main script to generate data and figures for the active learning experiment (Figure 1 and Section C.4 of the paper).
  - Performs the following tasks:
    - Generates 2D probability density plots (Gaussian and Christoffel densities).
    - Conducts the phase transition experiment to evaluate the success rate of sparse recovery under different sampling schemes (Christoffel Sampling (CS) and Monte Carlo (MC)).
    - Saves the resulting plots as EPS files.

- **generate_index_set.m**
  - Computes the (isotropic) tensor product, total degree, or hyperbolic cross multi-index set.
  - Inputs:
    - `index_type`: Either 'TP' (tensor product), 'TD' (total degree), or 'HC' (hyperbolic cross).
    - `d`: Dimension.
    - `n`: Polynomial order.
    - `verbose`: 0 (not verbose) or 1 (verbose).
  - Output:
    - `I`: The `d x n` array where the columns are the multi-indices from the desired multi-index set.

- **generate_hermite_matrix.m**
  - Generates a measurement matrix using tensor Hermite polynomials from an arbitrary multi-index set and collection of sample points.
  - Inputs:
    - `I`: `d x N` array of multi-indices.
    - `y_grid`: `m x d` array of sample points.
  - Output:
    - `A`: Normalized measurement matrix.

- **hermmat.m**
  - Generates the 1D matrix of Hermite polynomials.
  - Inputs:
    - `grid`: A column vector of points.
    - `k`: The desired number of polynomials to use.
  - Outputs:
    - `A`: The matrix of the first `k` Hermite polynomials evaluated on the grid.

- **find_order.m**
  - Computes the order of the largest multi-index set of a given type within a specific maximum size `N`.
  - Inputs:
    - `index_type`: Either 'TP' (tensor product), 'TD' (total degree), or 'HC' (hyperbolic cross).
    - `d`: Dimension.
    - `N`: Maximum desired size.
  - Output:
    - `n`: The largest polynomial order so that the index set has size at most `N`.

### Generated Figures

- **CS_d2_Ndes250_K100000_trials50_fig.eps**
  - Phase transition plot for the Christoffel Sampling (CS) sampling scheme.

- **Christoffel_density.eps**
  - Contour plot of the Christoffel density.

- **Gaussian_density.eps**
  - Contour plot of the Gaussian density.

- **MC_d2_Ndes250_K100000_trials50_fig.eps**
  - Phase transition plot for the Monte Carlo (MC) sampling scheme.

## Usage

1. Ensure all MATLAB scripts are in the same directory.
2. Open MATLAB and navigate to this directory.
3. Run the main script:
   ```matlab
   active_learning_exp
4.  The script will generate and save the figures as EPS files in the same directory.

## Figures Description

-   **Gaussian_density.eps**:
    -   Displays the Gaussian probability density over a 2D grid.
-   **Christoffel_density.eps**:
    -   Displays the Christoffel density over a 2D grid.
-   **CS_d2_Ndes250_K100000_trials50_fig.eps**:
    -   Shows the phase transition plot for the CS sampling scheme. The plot illustrates the success rate of sparse recovery as a function of the measurement-to-signal ratio (`m/N`) and sparsity-to-signal ratio (`s/N`).
-   **MC_d2_Ndes250_K100000_trials50_fig.eps**:
    -   Shows the phase transition plot for the MC sampling scheme. Similar to the CS plot, it visualizes the success rate of sparse recovery but using uniform random sampling instead.

## Notes

-   The phase transition experiment involves generating a random error grid, constructing measurement matrices, and evaluating the sparse recovery performance over multiple trials.
-   The SPGL1 solver is used for sparse recovery, with specified tolerance parameters to control the reconstruction accuracy.
-   The script outputs intermediate progress to the console, indicating the current sampling type, measurement size, sparsity level, trial number, and success status.

For any issues or further assistance, please refer to the paper or contact the authors.

