%--- Description ---%
%
% Filename: generate_data_approximations.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: Generates approximation error and condition number data for 
% the polynomial regression experiments

clear all; close all; clc;
addpath(genpath('../utils'))

d_vals = [1 2 4 8]; % values of d to use
p_ranges = [5 10 345 ; 5 5 100 ; 4 4 60 ; 2 2 22]; % values of p for the hyperbolic cross index set
N = 50000; % grid size
num_trials = 25; % number of trials
fun_name = 'iso_exp'; % function to use
samp_list = {'MCS','CS'}; % use MCS and CS strategies
poly_type = 'Hermite'; % polynomials to use: either 'Legendre' or 'Hermite'

for samp_type = samp_list
    
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
        
        % compute n values
        n_vals = [];
        for p = p_vals
            n_vals = [n_vals size(generate_index_set('HC',d,p),2)];
        end
        
        % compute m values
        m_vals = ceil(max(n_vals,n_vals.*log(n_vals))/(d+1));
        
        % generate data
        compute_approximations(char(poly_type),char(samp_type),char(fun_name),d,p_vals,m_vals,num_trials,err_grid);
        u = u+1;
        
    end
end
