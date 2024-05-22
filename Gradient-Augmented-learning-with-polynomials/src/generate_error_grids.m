%--- Description ---%
%
% Filename: generate_error_grids.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: Generates the error grids 

clear all; close all; clc;

d_vals = [1 2 4 8]; % values of d to use
N = 50000; % grid size
poly_type = 'Legendre'; % polynomials to use: either 'Legendre' or 'Hermite'

rng(1); % fix seed to 1

for d = d_vals
    
    % initialize file name
    file_name = ['error_grid','_d',num2str(d),'_N',num2str(N)];
    
    % generate error grid
    if isequal(poly_type,'Legendre')
        err_grid = 2*rand(N,d)-1; 
        file_name = ['Legendre_',file_name];
    else
        err_grid = randn(N,d);
        file_name = ['Hermite_',file_name];
    end
    
    % save error grid
    save(['../data/',file_name,'.mat'],'err_grid');
    
end