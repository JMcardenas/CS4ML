 %-------------------------------------------------------------------------%
% Filename: figs_App_D_data.m 
% Authors: Juan M. Cardenas .
% Part of the paper "CS4ML: A general sampling framework for Machine 
% Learning".
%
% Description: generates and saves the data for Figures Appendix D.
%-------------------------------------------------------------------------%
% Inputs:
% fig_num - figure number (3)
% row_num - row number (1)
% col_num - column number (either 1 or 2)
%-------------------------------------------------------------------------%

function fig_App_D_data(fig_num,row_num,col_num)
 
% Define name
fig_name = ['fig_AppD','_',num2str(fig_num),'_',num2str(row_num),'_',num2str(col_num)];

% Define parameters
nb_trials = 20;
nb_iter   = 9;
num_arch  = 2;
nb_samp   = 2;

if fig_num == 2
    nb_pts = 8010;
else
    nb_pts = 8000;
end

%%% Main loop %%%
k = 0;

if col_num == 1   
    y_values_data = zeros(nb_iter, nb_trials, num_arch*2);
    x_values_data = zeros(nb_iter, nb_trials, num_arch*2);
elseif col_num == 2 
    y_values_data = zeros(nb_pts, nb_trials);
    x_values_data = zeros(nb_pts, nb_trials);
else
    disp('incorrect column number: try 1 or 2');
end

if col_num == 1 
    for case_num = 1:num_arch  
            
        % load parameters
        load_param_fig_App_D
                                      
        disp(filename)
        load([filename '_extracted_data.mat']);
        
        % compute total m vals
        m_total_save_data = M_bound_values_save_data + M_col_values_save_data + M_init_values_save_data;
            
        % save data for samp method and trial
        for samp_num = 1:nb_samp
            for t = 1:nb_trials
                x_values_samp(:,t,samp_num) = l2_error_save_data(samp_num,t,2:end); 
                y_values_samp(:,t,samp_num) = m_total_save_data(samp_num,t,:);
            end
        end
        
        % save full data 
        y_values_data(:,:, case_num+k:case_num+k+1) = x_values_samp;    
        x_values_data(:,:, case_num+k:case_num+k+1) = y_values_samp;
        k = k+1;
    end

elseif col_num == 2 
    
    case_num = 1;
    samp_num = 1;
    
    % load parameters
    load_param_fig_App_D
                                      
    disp(filename)
    load([filename '_extracted_data.mat']);

    % save data for samp method and trial
    for t = 1:nb_trials
        x_values_data(:,t) = col_samples_save_data(samp_num,t,:,1); 
        y_values_data(:,t) = col_samples_save_data(samp_num,t,:,2);
    end

elseif col_num == 3
    
    case_num = 2;
    samp_num = 1;

    % load parameters
    load_param_fig_App_D
                                      
    disp(filename)
    load([filename '_extracted_data.mat']);
    
    % save data for samp method and trial
    for t = 1:nb_trials
        x_values_data(:,t) = col_samples_save_data(samp_num,t,:,1); 
        y_values_data(:,t) = col_samples_save_data(samp_num,t,:,2);
    end
else 
    disp(['incorrect colum number: try 1 or 2']);
end

%%% Save data %%%
save(['../../data/Figure App D/',fig_name], 'x_values_data', 'y_values_data');

end