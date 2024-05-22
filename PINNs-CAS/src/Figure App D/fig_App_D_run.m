%-------------------------------------------------------------------------%
% Filename: figs_App_D_run.m 
% Author: Juan M. Cardenas 
% Part of the paper "CS4ML: A general sampling framework for Machine 
% Learning".
% Description: generates the plots for Figure 3.
%-------------------------------------------------------------------------%

clear all; close all; clc;
addpath(genpath('../utils'));

for fig_num = 1
    
    if fig_num == 1
        row_init = 1; row_end = 1;
    else
        row_init = 1; row_end = 2;
    end

    for row_num = row_init:row_end
        for col_num = 1:3
             
            disp(' ');
            disp(['------------------------------------------------------------------------']);
            disp(['Running Figure App D',num2str(fig_num),'_',num2str(row_num),'_',num2str(col_num)]);
            disp(['------------------------------------------------------------------------']);
            disp(' ');
            fig_App_D_data(fig_num,row_num,col_num); 
    
        end
    end
end