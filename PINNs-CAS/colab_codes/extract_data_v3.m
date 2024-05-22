key = ''; % key to prepend file names (for separating runs)

% parameters
run_data_opt      = 1;
run_test_data_opt = 1;

% Set architecture
blocktype       = 'default';
layers          = 3; 
nodes_per_layer = 30;
tol             = '5e-07';
opt             = 'Adam';
initializer     = 'normal';
nb_schedule_epochs = 'doublein';
lrn_rate_schedule  = 'exp_decay';

% run IDs
unique_run_ID   = 'CedarPINNs_set1_3rd_arch_May1';
unique_run_ID_2 = [];

name_run_ID     = unique_run_ID;
unique_run_ID_3 = []; 

% directories
activations     = ["relu", "tanh", "elu", "sigmoid"];
init_act_num    = 2;
final_act_num   = 2;

% Running parameters
num_trials = 20;
num_steps  = 10;

if (layers == 5) || (layers == 10)
    nb_col_samples   = 8000;
    nb_init_samples  = 1000;
    nb_bound_samples = 1000;
elseif (layers == 3)
    nb_col_samples   = 8010;
    nb_init_samples  = 990;
    nb_bound_samples = 990;
else
    disp('incorrect number of layers, try 3, 5, or 10');
end

nb_hist_loss_pts  = 225009;
nb_plot_col_pts   = 10000;
nb_plot_init_pts  = 102;
nb_plot_bound_pts = 202;

dim_vals      = [1 2 4 8 16];
example_vals  = [1 2 3 5 6];

for ex_num = 1 
    
    example_num = example_vals(ex_num);

    for d_val = 2
        
        dim = dim_vals(d_val);

        switch dim
            case 1
                points = 10000
                nb_test_points = 16000 
            case 2
                points = 16000
                nb_test_points = 16000 
            case 4
                points = 20000
            case 8
                points = 50000
            case 16
                points = 100000
            otherwise
                disp('incorrect dim')
        end

        for i = init_act_num:final_act_num
            
            activation = convertStringsToChars(activations(i));

            %TODO: change the base_dir for your own case
            base_dir = '~/scratch/DNN_sampling/Results_PINNs_CAS';
            run_ID = [activation '_' blocktype '_' num2str(layers) 'x',...
                      num2str(nodes_per_layer) '_' num2str(points,'%06.f') '_pnts_' tol,...
                      '_tol_' opt '_opt_' nb_schedule_epochs '_schedule_epochs_',...
                      'burgers_example_' num2str(example_num) '_dim_' num2str(dim)]
            
            if isempty(unique_run_ID_2) == 0
                %TODO: delete this later, it's purpose is put together different unique_ID runs 
                run_ID_2 = [activation '_' blocktype '_' num2str(layers) 'x',...
                            num2str(nodes_per_layer) '_' num2str(points,'%06.f') '_pnts_' tol,...
                            '_tol_' opt '_opt_' nb_schedule_epochs '_schedule_epochs_',...
                            'burgers_example_' num2str(example_num) '_dim_' num2str(dim)]
            else 
                run_ID_2 = [];
            end

            samp_mode_dir_names = ["CAS",...
                                   "MC"];

            samp_mode_names = ["CAS",...
                               "MC"];

            num_modes = size(samp_mode_names,1);

            final_run_ID = [ unique_run_ID '_' activation '_' blocktype '_' num2str(layers) 'x',...
                            num2str(nodes_per_layer) '_' num2str(points,'%06.f') '_pnts_' tol,...
                            '_tol_' opt '_opt_' nb_schedule_epochs '_schedule_epochs_',...
                            'burgers_example_' num2str(example_num) '_dim_' num2str(dim)]

            output_filename = [final_run_ID '_extracted_data.mat'];

            extracted_data        = matfile(output_filename,'Writable',true);
            l2_error_save_data    = zeros(num_modes, num_trials, num_steps);
            
            if run_data_opt 
                
                training_loss_save_data  = zeros(num_modes, num_trials, nb_hist_loss_pts);
                l2_error_save_data       = zeros(num_modes, num_trials, num_steps);

                M_col_values_save_data   = zeros(num_modes, num_trials, num_steps-1);
                M_init_values_save_data  = zeros(num_modes, num_trials, num_steps-1);
                M_bound_values_save_data = zeros(num_modes, num_trials, num_steps-1);

                r_col_values_save_data   = zeros(num_modes, num_trials, num_steps-1);
                r_init_values_save_data  = zeros(num_modes, num_trials, num_steps-1);
                r_bound_values_save_data = zeros(num_modes, num_trials, num_steps-1);

                col_samples_save_data    = zeros(num_modes, num_trials, nb_col_samples, dim);
                init_samples_save_data   = zeros(num_modes, num_trials, nb_init_samples, dim);
                bound_samples_save_data  = zeros(num_modes, num_trials, nb_bound_samples, dim);
                
                nb_epochs_vec_save_data  = zeros(num_modes, num_trials, num_steps-1);
            
            else
                disp('run_data_opt and run_test_data_opt are not selected');
            end
                
            if run_test_data_opt
                
                model_plot_save_data           = zeros(num_modes, num_trials, 102, 101);
                Chris_plot_col_fun_save_data   = zeros(num_modes, num_trials, nb_plot_col_pts, 1);
                Chris_plot_init_fun_save_data  = zeros(num_modes, num_trials, nb_plot_init_pts, 1);
                Chris_plot_bound_fun_save_data = zeros(num_modes, num_trials, nb_plot_bound_pts, 1);
                
                Prob_plot_col_dist_save_data    = zeros(num_modes, num_trials, nb_plot_col_pts);
                Prob_plot_init_dist_save_data   = zeros(num_modes, num_trials, nb_plot_init_pts);
                Prob_plot_bound_dist_save_data  = zeros(num_modes, num_trials, nb_plot_bound_pts);
            
            else
                disp('run_data_opt and run_test_data_opt are not selected');
            end
            
            for samp_mode = 1:2

                samp_mode_name     = convertStringsToChars(samp_mode_names(samp_mode));
                samp_mode_dir_name = convertStringsToChars(samp_mode_dir_names(samp_mode));
                
                if isempty(run_ID_2) == 1
                    results_dir = [base_dir '/' unique_run_ID '/' run_ID '_training_method_' samp_mode_dir_name] 
                
                else 
                    % TODO: delete this later 
                    if samp_mode == 1 
                        % CAS sampling
                        results_dir = [base_dir '/' run_ID '_training_method_' samp_mode_dir_name] 
                    elseif samp_mode == 2
                        % MC sampling
                        results_dir = [base_dir '/' run_ID_2 '_training_method_' samp_mode_dir_name] 
                    else
                        disp('wrong samp mode')
                    end
                end

                for t = 1:num_trials

                    filename = ['trial_' num2str(t-1)];
                    
                    filename_train = [results_dir '/' filename '/run_data.mat'] 
                    filename_test  = [results_dir '/' filename '/run_test_data.mat']

                    if run_data_opt 
                        
                        data_train = matfile(filename_train,'Writable',false);
                        
                        if isfile(filename_train)

                            training_loss_save_data(samp_mode,t,:)  = data_train.hist_loss;
                            l2_error_save_data(samp_mode,t,:)       = data_train.l2_error_u_test;
                            nb_epochs_vec_save_data(samp_mode,t,:)  = data_train.nb_epochs_vec;

                            M_col_values_save_data(samp_mode,t,:)   = data_train.M_col_values;
                            M_init_values_save_data(samp_mode,t,:)  = data_train.M_init_values;
                            M_bound_values_save_data(samp_mode,t,:) = data_train.M_bound_values;

                            r_col_values_save_data(samp_mode,t,:)   = data_train.r_col_vals;
                            r_init_values_save_data(samp_mode,t,:)  = data_train.r_init_vals;
                            r_bound_values_save_data(samp_mode,t,:) = data_train.r_bound_vals;
                            
                            col_samples_save_data(samp_mode,t,:,:)   = data_train.col_samples;
                            init_samples_save_data(samp_mode,t,:,:)  = data_train.init_samples;
                            bound_samples_save_data(samp_mode,t,:,:) = data_train.bound_samples;
                        
                        else 
                            disp([filename_train ' is missing']); 
                        end
                    end

                    if run_test_data_opt
                                
                        data_test = matfile(filename_test,'Writable',false);
                        
                        if isfile(filename_test)
                            
                            model_plot_save_data(samp_mode,t,:,:)           = data_test.model_plot_data;
                            
                            Chris_plot_col_fun_save_data(samp_mode,t,:,:)   = data_test.Chris_plot_col_fun;
                            Chris_plot_init_fun_save_data(samp_mode,t,:,:)  = data_test.Chris_plot_init_fun;
                            Chris_plot_bound_fun_save_data(samp_mode,t,:,:) = data_test.Chris_plot_bound_fun;

                            Prob_plot_col_dist_save_data(samp_mode,t,:,:)   = data_test.Prob_plot_col_dist;
                            Prob_plot_init_dist_save_data(samp_mode,t,:,:)  = data_test.Prob_plot_init_dist;
                            Prob_plot_bound_dist_save_data(samp_mode,t,:,:) = data_test.Prob_plot_bound_dist;

                        else
                            disp([filename_test ' is missing']);
                        end
                    end
                end
            end
            
            if run_data_opt
                
                extracted_data.training_loss_save_data  = training_loss_save_data;
                extracted_data.l2_error_save_data       = l2_error_save_data;
                extracted_data.nb_epochs_vec_save_data  = nb_epochs_vec_save_data;    

                extracted_data.M_col_values_save_data   = M_col_values_save_data;
                extracted_data.M_init_values_save_data  = M_init_values_save_data;
                extracted_data.M_bound_values_save_data = M_bound_values_save_data;

                extracted_data.r_col_values_save_data   = r_col_values_save_data;    
                extracted_data.r_init_values_save_data  = r_init_values_save_data;    
                extracted_data.r_bound_values_save_data = r_bound_values_save_data;    

                extracted_data.col_samples_save_data    = col_samples_save_data;
                extracted_data.init_samples_save_data   = init_samples_save_data;
                extracted_data.bound_samples_save_data  = bound_samples_save_data;

            else
                disp('incorrect run_data option');
            end
            
            if run_test_data_opt

                extracted_data.model_plot_save_data           = model_plot_save_data;    

                extracted_data.Chris_plot_col_fun_save_data   = Chris_plot_col_fun_save_data;
                extracted_data.Chris_plot_init_fun_save_data  = Chris_plot_init_fun_save_data;
                extracted_data.Chris_plot_bound_fun_save_data = Chris_plot_bound_fun_save_data;

                extracted_data.Prob_plot_col_dist_save_data   = Prob_plot_col_dist_save_data;
                extracted_data.Prob_plot_init_dist_save_data  = Prob_plot_init_dist_save_data;
                extracted_data.Prob_plot_bound_dist_save_data = Prob_plot_bound_dist_save_data;

            else
                disp('incorrect run_data option');
            end

        end
    end
end
