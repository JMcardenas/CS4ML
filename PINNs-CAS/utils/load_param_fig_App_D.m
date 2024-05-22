%-------------------------------------------------------------------------%
% Filename: load_param_fig_App_D.m 
% Authors: Juan M. Cardenas .
% Part of the paper "CS4ML: A general sampling framework for Machine 
% Learning".
%
% Description: generates parameters for Figures Appendix D.
%-------------------------------------------------------------------------%
% Inputs:
% fig_num - figure number (either 1, 2, or 3)
% row_num - row number (either 1, 2 or 3)
% col_num - column number (either 1, 2 or 3)
%-------------------------------------------------------------------------%

% pre-plot parameters
example_num = 1;        % set example number 
dim = 2;                % set input dim 
points = 16000;         % set points 

% set hyperparameters 
setdate     = 'May1';
setinit     = 'normal';
blocktype   = 'default';
tol         = '5e-07';
opt         = 'Adam';

% activation names
activations              = ["ReLU", "$\tanh$", "ELU", "sigmoid"];
activations_dirnames     = ["relu", "tanh", "elu", "sigmoid"];

% initialization
initializations          = ["normal", "he_normal"];
initializations_dirnames = ["normal", "He normal"];

% lrn rate
lrn_rate                 = ["exp", "const"];
lrn_rate_dirnames        = ["exp", "const"];

% dirnames
set_dirnames    = ["1st", "2nd", "3rd", "4th"]; 
schedule_epochs = ["constant", "midlein", "doublein"];

% layers and nodes
arch_layers = [1 2 3 4 5 10 20 30];
arch_nodes  = [10 20 30 40 50 100 200 300];
 
% load parameters 
if fig_num == 1  

    arch_num       = 5;           % set nb layers and nodes
    sche_epoch_num = 3;           % set schedule epoch name
    setdir_num     = 1;           % set dirname  

    if case_num == 1 
        act_num = 1;           % set activation
        set_num = 1;           % set setnumber 1 or 2  
    elseif case_num == 2          
        act_num = 3;     
        set_num = 2; 
    else
        disp(['incorrect case_num: use 1 or 2']);
    end

elseif fig_num == 2

    arch_num       = 3;           % set nb layers and nodes
    sche_epoch_num = 3;           % set schedule epoch name
    setdir_num     = 3;           % set dirname  
    
    if row_num == 1
        if case_num == 1 
            act_num = 2;           % set activation
            set_num = 1;           % set setnumber 1 or 2 
        elseif case_num == 2          
            act_num = 4;          
            set_num = 2; 
        else
            disp(['incorrect case_num: use 1 or 2']);
        end

    elseif row_num == 2
        if case_num == 1 
            act_num = 1;           % set activation    
            set_num = 1;           % set setnumber 1 or 2  
        elseif case_num == 2          
            act_num = 3;         
            set_num = 2; 
        else
            disp(['incorrect case_num: use 1 or 2']);
        end    
    else
        disp(['incorrect case_num: use 1 or 2']);
    end

elseif fig_num == 3

    arch_num       = 6;           % set nb layers and nodes
    sche_epoch_num = 3;           % set schedule epoch name
    setdir_num     = 2;           % set dirname  

    if row_num == 1 
        if case_num == 1 
            act_num = 2;           % set activation
            set_num = 1;           % set setnumber 1 or 2 
        elseif case_num == 2          
            act_num = 4;          
            set_num = 2; 
        else
            disp(['incorrect case_num: use 1 or 2']);
        end

    elseif row_num == 2 
        if case_num == 1 
            act_num = 1;           % set activation    
            set_num = 1;           % set setnumber 1 or 2  
        elseif case_num == 2          
            act_num = 3;         
            set_num = 2; 
        else
            disp(['incorrect case_num: use 1 or 2']);
        end    
    else
        disp(['incorrect case_num: use 1 or 2']);
    end    
else
    disp(['incorrect case_num: use 1 or 2']);
end

% construct filename
home_dir  = ['../../results']; 
setdir    = convertStringsToChars(set_dirnames(setdir_num));
base_dir  = ['CedarPINNs_set' num2str(set_num) '_' setdir '_arch_' setdate];
                               
act_name   = convertStringsToChars(activations_dirnames(act_num)); 
arch_name  = [num2str(arch_layers(arch_num)) 'x' num2str(arch_nodes(arch_num))];             
epoch_name = convertStringsToChars(schedule_epochs(sche_epoch_num)); 
    
filename = [home_dir '/' base_dir '_' act_name '_' blocktype '_' arch_name '_'...
            num2str(points,'%06.f') '_pnts_' tol '_tol_' opt '_opt_' epoch_name...
            '_schedule_epochs_' 'burgers_example_' num2str(example_num) '_dim_' num2str(dim)]; 
