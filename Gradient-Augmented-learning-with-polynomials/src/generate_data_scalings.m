%--- Description ---%
%
% Filename: generate_data_approximations.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: Generates m versus n scaling data for the polynomial 
% regression experiments

clear all; close all; clc;
addpath(genpath('../utils'))

d_vals = [1 2 4 8]; % values of d to use

N = 50000; % error grid size
num_trials = 25; % number of trials
tol = 10; % tolerance
samp_list = {'MCS','CS'}; % use Monte Carlo or the near-optimal sampling strategy
poly_type = 'Hermite'; % polynomials to use: either 'Legendre' or 'Hermite'

% values of p for the hyperbolic cross index set
p_ranges_1 = [20 20 300 ; 7 7 105 ; 3 3 45 ; 1 1 16];
p_ranges_2 = [2 2 18 ; 1 1 12 ; 1 1 11 ; 1 1 9];
p_ranges_3 = [20 20 180 ; 7 7 105 ; 3 3 45 ; 1 1 16];

%% compute average scalings
for samp_type = samp_list
    
    if isequal(char(samp_type),'MCS') && isequal(char(poly_type),'Legendre')
        p_ranges = p_ranges_3;
    elseif isequal(char(samp_type),'MCS') && isequal(char(poly_type),'Hermite')
        p_ranges = p_ranges_2;
    else
        p_ranges = p_ranges_1;
    end

    u = 1;
    for d = d_vals
        
        % specify p values
        p_vals = p_ranges(u,1):p_ranges(u,2):p_ranges(u,3);
        
        % set file name
        file_name = ['error_grid','_d',num2str(d),'_N',num2str(N)];
        if isequal(poly_type,'Legendre')
            file_name = ['Legendre_',file_name];
        else
            file_name = ['Hermite_',file_name];
        end
        
        % load error grid
        load(['../data/',file_name,'.mat'])
        
        % compute scalings
        compute_scalings(char(poly_type),char(samp_type),d,p_vals,num_trials,err_grid,tol);
        u = u+1;
        
    end
end
