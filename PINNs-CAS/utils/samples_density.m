%-------------------------------------------------------------------------%
% Description: Generate plot Christoffel function related to burgers eq.
% Author: Juan M. Cardenas
%-------------------------------------------------------------------------%
clear all; close all; clc;
%-------------------------------------------------------------------------%
% load Christoffel function
fig_folder  = '~/Downloads/Burgers equation solution/Figures';

folder_name_1 = ['/home/juan/Documents/DNN ASGD/PINNs - CAS/results' ...
              '/ColabPINNs_set2_1st_arch_Apr11/sigmoid_default_5x50_'...
              '016000_pnts_5e-07_tol_Adam_opt_constant_schedule_epochs_'...
              'burgers_example_1_dim_2_training_method_CAS/trial_0'];
file_name_1  = [folder_name_1 '/run_test_data.mat'];
file_name_2  = [folder_name_1 '/run_data.mat'];

load(file_name_1)  
load(file_name_2)
%-------------------------------------------------------------------------%
% load T and X grid 
file_name_3 = ['/home/juan/Downloads/Burgers equation solution/data'...
               '/plot_data_example_1_dim_2.mat'];
                
load(file_name_3)
%-------------------------------------------------------------------------%
% set parameters
plot_option = 2;

%-------------------------------------------------------------------------%

if plot_option == 1

    % Plot points over solution 
    fig1 = figure();
    ms = 5;
   
    addpath('~/Downloads/matlab - downloads/')
         
    plot(col_samples(:,1),col_samples(:,2),'b.','MarkerSize',ms) 
    
    title('$u$-sol. and samples drawn by CS', 'Interpreter','latex')
    
    namefig = 'sol_and_CS_samples_data_opt1';
    saveas(fig1, fullfile(fig_folder, namefig),'epsc');

elseif plot_option == 2
    
    % Create a chart where transparency varies with density:
    fig2 = densityScatterChart(col_samples(:,1),col_samples(:,2), 'UseColor', true, 'UseAlpha', true);
    
    fig2.DensityMethod = 'ksdensity';
    fig2.AlphaRange    = [.2 .5];
    
    namefig = 'sol_and_CS_samples_data_opt2';
    saveas(fig2, fullfile(fig_folder, namefig),'epsc');
else
    dips('incorrect option select 1 or 2');
end

