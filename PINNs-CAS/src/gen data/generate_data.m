%-------------------------------------------------------------------------%
% Description: This script generates data for burgers equation  
%-------------------------------------------------------------------------%
clear all; close all; clc;
%-------------------------------------------------------------------------%

addpath(genpath('../../utils'));

% Parameters
nb_example  = 1;          % number of example 
input_dim   = 2;          % t and x 
nu          = 0.01/pi;    % viscocity 
data_type   = "uniform";  % uniform or equispaced

% set domain 
if nb_example == 1
    % D = [-1,1] x I = [0,1]
    x_val_init  = -1;   x_val_final = 1;
    t_val_init  = 0;    t_val_final = 1;
elseif nb_example == 2
    % D = [0, 2*pi] x I = [0,1]
    x_val_init  = 0;    x_val_final = 2*pi;
    t_val_init  = 0;    t_val_final = 1;
else
    disp('incorrect number of example: use 1 or 2')
end

% set treshold number
tresh_num   = (x_val_final + x_val_init)/2;

%-------------------------------------------------------------------------%
% iterate over train, test data 
for i = 1:2
    % set number of pts 
    if i == 1
        % training data 
        disp('-------------------------------------------------------------')
        disp('Generating Training Data');

        if convertStringsToChars(data_type) == "uniform"
            % uniform
            nb_init_data  = 2000; 
            nb_bound_data = 2000;
            nb_x_col_data = 16000;
            nb_t_col_data = nb_x_col_data; 
        else
            %equispaced
            nb_init_data  = 10; 
            nb_bound_data = 10;
            nb_x_col_data = 100;
            nb_t_col_data = nb_x_col_data; 
        end

    else
        % testing data 
        disp('-------------------------------------------------------------')
        disp('Generating Testing Data')
    
        if convertStringsToChars(data_type) == "uniform"
            %uniform
            nb_init_data  = 1000; 
            nb_bound_data = 1000;
            nb_x_col_data = 8000;
            nb_t_col_data = nb_x_col_data; 
        else
            %equispaced
            nb_init_data  = 10; 
            nb_bound_data = 10;
            nb_x_col_data = 100;
            nb_t_col_data = nb_x_col_data; 
        end  
    end
    
    disp('-------------------------------------------------------------');
    disp(['number of samples for inital data:      ', num2str(nb_init_data)]);
    disp(['number of samples for boundary data:    ', num2str(nb_bound_data)]);
    disp(['number of samples for collocation data: ', num2str(nb_x_col_data)]) 
    disp('-------------------------------------------------------------');
    disp(['Domain [x_min,x_max] x [0,T]: [', num2str(x_val_init), ',',...
          num2str(x_val_final), '] x [0,', num2str(t_val_final), ']']);
    disp('-------------------------------------------------------------');
    
    if convertStringsToChars(data_type) == "uniform"
        % collocation data 
        x_col_data = x_val_init + (x_val_final - x_val_init).*rand(nb_x_col_data,1);
        t_col_data = t_val_init + (t_val_final - t_val_init).*rand(nb_t_col_data,1);
    
        % initial data 
        x_init_data = x_val_init + (x_val_final - x_val_init).*rand(nb_init_data,1); 
        t_init_data = zeros(nb_init_data,1);
        
        % boundary data 
        x_bound_data = x_val_init + (x_val_final - x_val_init).*rand(nb_bound_data,1);
        t_bound_data = t_val_init + (t_val_final - t_val_init).*rand(nb_bound_data,1);

        % search treshold set 
        I_tre  = (x_bound_data >= tresh_num);
        I_com  = (x_bound_data < tresh_num);
            
        if nb_example == 1 
            x_bound_data(I_tre) = 1;
            x_bound_data(I_com) = -1;
        
        else
            x_bound_data(I_tre) = 2*pi;
            x_bound_data(I_com) = 0;
        end
        
        % rearrange all data
        x_data = [x_col_data; x_init_data; x_bound_data];
        t_data = [t_col_data; t_init_data; t_bound_data];
     
        % number of points
        nb_pts = nb_x_col_data + nb_init_data + nb_bound_data;
        u_data = zeros(nb_pts,1);

        for k = 1:nb_pts
            % set pts 
            vx_k = x_data(k);
            vt_k = t_data(k);

            if nb_example == 1
                u_data(k,1) = burgers_viscous_time_exact1( nu, 1, vx_k, 1, vt_k); 
            else
                u_data(k,1) = burgers_viscous_time_exact2( nu, 1, vx_k, 1, vt_k);
            end
        end

        % set indices
        ind_init_min  = nb_x_col_data + 1;
        ind_init_max  = nb_x_col_data + nb_init_data;
        ind_bound_min = nb_pts - nb_bound_data + 1;
        ind_bound_max = nb_pts;
    
        u_col_data   = u_data(1:nb_x_col_data,1);
        u_init_data  = u_data(ind_init_min:ind_init_max);
        u_bound_data = u_data(ind_bound_min:ind_bound_max);
    
    else 
        % equispaced data 
        x_data = linspace(x_val_init,x_val_final,nb_x_col_data);
        t_data = linspace(t_val_init,t_val_final,nb_t_col_data); 
        nb_pts = nb_t_col_data*nb_x_col_data; 
        
        [x_data_sort, ~] = sort(x_data);
        [t_data_sort, ~] = sort(t_data);
            
        if nb_example == 1
            u_data = burgers_viscous_time_exact1( nu, nb_x_col_data, x_data_sort, nb_t_col_data, t_data_sort);
        else
            u_data = burgers_viscous_time_exact2( nu, nb_x_col_data, x_data_sort, nb_t_col_data, t_data_sort);
        end
             
        nb_x_data = length(x_data_sort);
        nb_t_data = length(t_data_sort); 
    end 
    %---------------------------------------------------------------------%
    % save train, test data 
    if i == 1
        name_file = ['train_data_example_' num2str(nb_example)...
                     '_dim_' num2str(input_dim) '.mat'];

        save(['../../colab_data/' name_file],...
             't_col_data', 'x_col_data', 'u_col_data',...
             't_init_data', 'x_init_data', 'u_init_data',...
             't_bound_data', 'x_bound_data', 'u_bound_data',...
             'nb_pts','nb_x_col_data', 'nb_init_data', 'nb_bound_data');
    else
        name_file = ['test_data_example_' num2str(nb_example)...
                     '_dim_' num2str(input_dim) '.mat'];

        %%% Save data %%%
        save(['../../colab_data/' name_file],...
             't_data', 'x_data', 'u_data',...
             'nb_t_col_data', 'nb_x_col_data',...
             'nb_bound_data','nb_init_data','nb_pts');
    end
     
    disp(['Data saved in folder: ' '../../colab_data/']);
    disp(['name file: ' name_file]);
    disp('-------------------------------------------------------------');
    %---------------------------------------------------------------------%
    % Plot grid
    fig_grid = figure();
    
    plot(t_col_data, x_col_data,'.','MarkerSize',1); hold on;
    plot(t_init_data, x_init_data, '.','MarkerSize',1); hold on;
    plot(t_bound_data, x_bound_data, '.', 'MarkerSize',1); 
    
    if i == 1
        name_title = 'training';
    else
        name_title = 'testing';
    end
    
    xlabel('$t$-values','Interpreter','latex')
    ylabel('$x$-values','Interpreter','latex')
    title(['grid values on ' name_title ' data'],'Interpreter','latex')

    % Plot solution 3D
    fig_sol = figure();
    
    plot3(t_col_data,x_col_data, u_col_data, '.','MarkerSize',1); hold on;
    plot3(t_init_data,x_init_data, u_init_data, '.','MarkerSize',1); hold on;
    plot3(t_bound_data,x_bound_data, u_bound_data, '.','MarkerSize',1); 

    if i == 1
        name_title = 'training';
    else
        name_title = 'testing';
    end
    title(['$u$-values on ' name_title ' data'],'Interpreter','latex')
    grid on 

    % solution in 2D
    grid off 
    view(2)    
end

%---------------------------------------------------------------------%
% Generate plot data  

nb_plot_x_data = 102;
nb_plot_t_data = 101;   

disp('Generating Plot Data');
disp('-------------------------------------------------------------'); 
disp(['number of samples for grid [t,x]:       ' num2str(nb_plot_t_data) 'x'...
       num2str(nb_plot_x_data)]); 

x_plot_data = linspace(x_val_init,x_val_final,nb_plot_x_data);
t_plot_data = linspace(t_val_init,t_val_final,nb_plot_t_data); 
nb_plot_pts = nb_plot_t_data*nb_plot_x_data; 

[x_plot_data_sort, I_x] = sort(x_plot_data);
[t_plot_data_sort, I_t] = sort(t_plot_data);

% compute solution
if nb_example == 1
    u_plot_data = burgers_viscous_time_exact1( nu, nb_plot_x_data, x_plot_data_sort, nb_plot_t_data, t_plot_data_sort);
else
    u_plot_data = burgers_viscous_time_exact2( nu, nb_plot_x_data, x_plot_data_sort, nb_plot_t_data, t_plot_data_sort);
end
        
% set boundary conditions
if nb_example == 1
    u_plot_data(1,:)   = 0;
    u_plot_data(end,:) = 0;
else
    u_plot_data(1,:) = u_plot_data(end,:);
end

nb_plot_x_data = length(x_plot_data_sort);
nb_plot_t_data = length(t_plot_data_sort);
    
[T,X]  = meshgrid(t_plot_data_sort,x_plot_data_sort);
Z_grid = [reshape(T, nb_plot_t_data*nb_plot_x_data,1), reshape(X, nb_plot_t_data*nb_plot_x_data,1)];

% collocation data 
t_plot_matrix_col_data = T(2:end-1,2:end);
x_plot_matrix_col_data = X(2:end-1,2:end);
u_plot_matrix_col_data = u_plot_data(2:end-1,2:end);

nb_plot_col_data = (nb_plot_t_data-1)*(nb_plot_x_data-2); 

t_plot_col_data = reshape(t_plot_matrix_col_data, (nb_plot_t_data-1)*(nb_plot_x_data-2),1);
x_plot_col_data = reshape(x_plot_matrix_col_data, (nb_plot_t_data-1)*(nb_plot_x_data-2),1);
u_plot_col_data = reshape(u_plot_matrix_col_data, (nb_plot_t_data-1)*(nb_plot_x_data-2),1);

% initial data 
t_plot_init_data = t_val_init*ones(nb_plot_x_data,1);
x_plot_init_data = x_plot_data';
u_plot_init_data = u_plot_data(:,1);

nb_plot_init_data = (1)*(nb_plot_x_data);

% bound data 
t_plot_bound_data = [t_plot_data';...
                     t_plot_data'];
x_plot_bound_data = [x_plot_data(1)*ones(nb_plot_t_data,1);...
                     x_plot_data(end)*ones(nb_plot_t_data,1)]; 
u_plot_bound_data = [u_plot_data(1,:); u_plot_data(end,:)]';

nb_plot_bound_data = (nb_plot_t_data)*(2);

% disp
disp(['number of samples for initial grid:     ' num2str(nb_plot_init_data)]);
disp(['number of samples for boundary grid:    ' num2str(nb_plot_bound_data)]);
disp(['number of samples for collocation grid: ' num2str(nb_plot_col_data)]);
disp('-------------------------------------------------------------');

% plot solution on grid 3D
fig1 = figure();
s = surf(T,X,u_plot_data);
s.EdgeColor = 'none';
title('$u$ solution on the domain $D\times I$', 'Interpreter','latex')
colorbar() 

if nb_plot_x_data > 1000
    s.EdgeColor = 'none';
end
     
% solution in 2D
grid off 
view(2) 

% plot solution on initial data 
fig2 = figure();
% plot solution on initial data
plot(x_plot_init_data, u_plot_data(:,1), 'b')
grid on 
title('$u$ solution on initial data','Interpreter','latex')
xlabel('$x$-values','Interpreter','latex')
ylabel('$u$-values at $t=0$','Interpreter','latex') 

% plot soloution on boundary data through time 
fig3 = figure();
plot(t_plot_bound_data(1:end/2), u_plot_data(1,:), 'r')
hold on
plot(t_plot_bound_data(1:end/2), u_plot_data(end,:), 'k')
grid on 
title('$u$ solution on boundary data','Interpreter','latex')
xlabel('$t$-values','Interpreter','latex')
ylabel('$u$-values at $t=0$','Interpreter','latex') 

name_file = ['plot_data_example_' num2str(nb_example) '_dim_' ...
              num2str(input_dim) '.mat'];
%%% Save data %%%
save(['../../colab_data/' name_file],...
    'T', 'X', 'Z_grid', 'u_plot_data',...
    't_plot_col_data', 'x_plot_col_data', 'u_plot_col_data',...
    't_plot_init_data', 'x_plot_init_data', 'u_plot_init_data',...
    't_plot_bound_data', 'x_plot_bound_data', 'u_plot_bound_data',...
    'nb_plot_col_data', 'nb_plot_init_data', 'nb_plot_bound_data',...
    'nb_plot_x_data', 'nb_plot_t_data', 'nb_plot_pts');

disp(['Data saved in folder: ' '../../colab_data/']);
disp(['name file: ' name_file]);
disp('-------------------------------------------------------------');