%--- Description ---%
%
% Filename: active_learning_exp.m
% Authors: Ben Adcock, Juan M. Cardenas, Nick Dexter
% Part of the paper "A Unified Framework for Learning with Nonlinear Model 
% Classes from Arbitrary Linear Samples"
%
% Description: Generates the data and figures for the active learning
% experiment (see Figure 1 and Section C.4)

clear all; close all; clc;

d = 2; % dimension
N_des = 250; % desired size of N
K = 100000; % error grid size
samp_list = {'CS','MC'}; % types of sampling to consider

%%% density plot parameters
xmin = -5;
xmax = 5;
num_pts = 200;

%%% SPGL1 parameters
eta = 0;
eps1 = 1e-4;
eps2 = 1e-5;
iterations = 1000;
opts = spgSetParms('verbosity',0,'iterations',iterations,'optTol',eps1,'bpTol',eps2); % specify spgl1 parameters

%%% Phase transition parameters
trials = 50;
err_tol = 1e-2;
num_m = 50;
num_s = 50;

%%% plotting parameters
lw = 2; % linewidth
fs = 24; % fontsize

%%% Construct total degree index set
n = find_order('TD',d,N_des); % find the largest TD index set of size <= N
I = generate_index_set('TD',d,n,0); % generate the HC index set
N = size(I,2); % actual size of N

%% Compute 2D probability densities and plot

grid = linspace(xmin,xmax,num_pts);
[X,Y] = meshgrid(grid,grid);

%%% compute Gaussian density
G = zeros(num_pts);
for i = 1:num_pts
    for j = 1:num_pts
        G(i,j) = exp(-(grid(i)^2+grid(j)^2)/2)/(2*pi);
    end
end

%%% compute Christoffel density
xy_grid = zeros(num_pts^2,2);
for i = 1:num_pts
    for j = 1:num_pts
        xy_grid(num_pts*(i-1)+j,:) = [grid(i) grid(j)];
    end
end

A = generate_hermite_matrix(I,xy_grid);
Ch = max(abs(A).^2,[],2);
Ch = reshape(Ch,[num_pts,num_pts]);
Ch = Ch.*G;

Ch = Ch/sum(Ch(:));
G = G/sum(G(:));

fig = figure(1);
contourf(X,Y,G)
colormap hot
ax = gca;
ax.FontSize = fs;
ax.LineWidth = lw;
ax.XTick = [-5 -2.5 0 2.5 5];
ax.YTick = [-5 -2.5 0 2.5 5];
xlabel('$x_1$','Interpreter','latex','FontSize',fs);
ylabel('$x_2$','Interpreter','latex','FontSize',fs);
set(gca, 'LooseInset', get(gca, 'TightInset'));
saveas(fig,'Gaussian_density','epsc');

fig = figure(2);
contourf(X,Y,Ch)
colormap hot
ax = gca;
ax.FontSize = fs;
ax.LineWidth = lw;
ax.XTick = [-5 -2.5 0 2.5 5];
ax.YTick = [-5 -2.5 0 2.5 5];
xlabel('$x_1$','Interpreter','latex','FontSize',fs);
ylabel('$x_2$','Interpreter','latex','FontSize',fs);
set(gca, 'LooseInset', get(gca, 'TightInset'));
saveas(fig,'Christoffel_density','epsc');

%% Run phase transition experiment

%%% Generate error grid and probability vectors
err_grid = randn(K,d); % Generate error grid
A_err_grid = generate_hermite_matrix(I,err_grid); % generate CS error matrix
[Q,~] = qr(A_err_grid,0);

for samp_type = samp_list
    
    file_name = [char(samp_type),'_d',num2str(d),'_Ndes',num2str(N_des),'_K',num2str(K),'_trials',num2str(trials)];
    
    if isequal(char(samp_type),'CS')
        prob = K*max(abs(Q).^2,[],2); % construct optimal sampling distribution
        CS_const = sum(prob)/K;
        MC_const = max(prob);
        disp(['The Christoffel sampling constant is ',num2str(CS_const)]);
        disp(['The Monte Carlo sampling constant is ',num2str(MC_const)]);
        
    else
        prob = ones(K,1);
    end
    prob = prob/sum(prob);
    
    m_vals = round(linspace(1,N,num_m));
    s_vals =round(linspace(1,N/2,num_m));
    
    PT_matrix = zeros(num_m,num_s);
    
    for i = 1:num_m
        
        m = m_vals(i);
        
        % generate measurement matrix
        L = datasample((1:K)',m,'Replace',true,'Weights',prob);
        W = diag(1./sqrt(K*prob(L)));
        x_grid = err_grid(L,:);
        A = sqrt(K/m)*W*Q(L,:);
        
        for j = 1:num_s
            
            s = s_vals(j);
            
            % do not run if s exceeds m
            if s > m
                break;
            end
            
            err_vals = zeros(trials,1);
            for t = 1:trials
                
                % Generate random x
                x = zeros(N,1);
                x(1:s) = randn(s,1);
                x = x(randperm(N));
                
                b = A*x; % Generate measurement vector b = A x
                xhat = spg_bpdn(A,b,eta,opts); % Reconstruct x
                err_vals(t) = norm(x-xhat)/norm(x); % Compute error
                
                disp([char(samp_type),' ','m = ',num2str(m),' s = ',num2str(s),' trial = ',num2str(t),' success = ',num2str(err_vals(t) <= err_tol)]);
                
            end
            
            PT_matrix(i,j) = sum(err_vals <= err_tol)/trials;
            
        end
        
    end
    
    fig = figure(3);
    imagesc(m_vals/N,s_vals/N,PT_matrix')
    ax = gca;
    ax.YDir = 'normal';
    ax.FontSize = fs;
    ax.LineWidth = lw;
    xlabel('$m/N$','Interpreter','latex','FontSize',fs);
    ylabel('$s/N$','Interpreter','latex','FontSize',fs);
    saveas(fig,[file_name,'_fig'],'epsc');
    
end
