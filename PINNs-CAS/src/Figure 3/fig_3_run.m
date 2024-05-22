%-------------------------------------------------------------------------%
% Filename: figs_3_run.m 
% Author: Juan M. Cardenas 
% Part of the paper "CS4ML: A general sampling framework for Machine 
% Learning".
% Description: generates the plots for Figure 3.
%-------------------------------------------------------------------------%

clear all; close all; clc;
addpath(genpath('../utils'));

for row_num = 1
    for col_num = 3
        
        fig_num = 3;
        disp(' ');
        disp(['------------------------------------------------------------------------']);
        disp(['Running Figure ',num2str(fig_num),'_',num2str(row_num),'_',num2str(col_num)]);
        disp(['------------------------------------------------------------------------']);
        disp(' ');
        fig_3_data(row_num,col_num); 

    end
end
