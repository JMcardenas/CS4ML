%--- Description ---%
%
% Filename: generate_error_grids.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: Plots and saves the figures

clear all; close all; clc;
addpath(genpath('../utils'))

d_vals = [1 2 4 8]; % values of d to use
N = 50000; % grid size
num_trials = 25; % number of trials
fun_name = 'iso_exp'; % function to use
samp_list = {'MCS','CS'}; % use MCS and CS strategies
poly_type = 'Hermite'; % polynomials to use: either 'Legendre' or 'Hermite'

% set various plotting parameters
[ms, lw, fs, colors, markers, AlphaLevel] = get_fig_param();

for i = 1:length(colors)
    new_colors{2*i-1} = colors{i};
    new_colors{2*i}   = 0.7 * colors{i};
end
new_markers = markers;
for i = 2:2:length(markers)
    new_markers{i} = ['-',markers{i}];
end
colors = new_colors;
markers = new_markers;

%% plot errors and condition numbers

p_ranges = [5 10 345 ; 5 5 100 ; 4 4 60 ; 2 2 22]; % values of p for the hyperbolic cross index set

u = 1;
for d = d_vals
    
    fig = figure(1);
    
    i_curve = 1; hplot = []; % initialize legend function handle
    
    for samp_type = samp_list
        
        % specify p values
        p_vals = p_ranges(u,1):p_ranges(u,2):p_ranges(u,3);
        
        % compute n values
        n_vals = [];
        for p = p_vals
            n_vals = [n_vals size(generate_index_set('HC',d,p),2)];
        end
        
        % compute m values
        m_vals = ceil(max(n_vals,n_vals.*log(n_vals)/(d+1)));
        
        % set file name
        file_name = [char(fun_name),'_',char(samp_type),'_d',num2str(d),'_N',num2str(N),'_trials',num2str(num_trials)];
        if isequal(poly_type,'Legendre')
            file_name = ['Legendre_',file_name];
        else
            file_name = ['Hermite_',file_name];
        end
        
        % load data
        load(['../data/',file_name,'.mat'])
        
        hplot = [hplot plot_book_style_each_curve(m_vals,err_data','shaded','mean_std_log10',i_curve)];
        i_curve = i_curve+1;
        
    end
    
    hold off
    set(gca, 'yscale', 'log','xscale','log');
    axis tight
    
    h = legend(hplot,'MCS','CS','location','northeast');
    
    xlabel('$m$','interpreter', 'latex')
    ylabel('test error $E(f)$','interpreter', 'latex')
    
    set_axis_param
    set_fonts
    
    ax = gca;
    Z = ax.XLim;
    Z(2) = max(Z(2),1000);
    ax.XLim = Z;
    ax.XTick = [10 100 1000];
    ax.XTickLabels = {'10^1', '10^2', '10^3'};
    
    if d == 1 || d == 2
        ax.YTick = [1e-15 1e-10 1e-5 1e0];
        ax.YTickLabels = {'10^{-15}','10^{-10}','10^{-5}','10^{0}'};
    elseif d == 4
        ax.YTick = [1e-9 1e-6 1e-3 1e0 1e3];
        ax.YTickLabels = {'10^{-9}','10^{-6}','10^{-3}','10^{0}','10^{3 }'};
    else
        ax.YTick = [1e-6,1e-4 1e-2 1e-0];
        ax.YTickLabels = {'10^{-6}','10^{-4}','10^{-2}','10^{0}'};
    end
    
    fig_name = ['err_d',num2str(d),'_N',num2str(N),'_trials',num2str(num_trials)];
    if isequal(poly_type,'Legendre')
        fig_name = ['Legendre_',fig_name];
    else
        fig_name = ['Hermite_',fig_name];
    end
    saveas(fig,['../figs/',fig_name],'epsc');
    
    u = u+1;
    
end

u = 1;
for d = d_vals
    
    fig = figure(1);
    
    i_curve = 1; hplot = []; % initialize legend function handle
    
    for samp_type = samp_list
        
        % specify p values
        p_vals = p_ranges(u,1):p_ranges(u,2):p_ranges(u,3);
        
        % compute n values
        n_vals = [];
        for p = p_vals
            n_vals = [n_vals size(generate_index_set('HC',d,p),2)];
        end
        
        % compute m values
        m_vals = ceil(max(n_vals,n_vals.*log(n_vals)/(d+1)));
        
        % set file name
        file_name = [char(fun_name),'_',char(samp_type),'_d',num2str(d),'_N',num2str(N),'_trials',num2str(num_trials)];
        if isequal(poly_type,'Legendre')
            file_name = ['Legendre_',file_name];
        else
            file_name = ['Hermite_',file_name];
        end
        
        % load data
        load(['../data/',file_name,'.mat'])
        
        hplot = [hplot plot_book_style_each_curve(m_vals,cond_data','shaded','mean_std_log10',i_curve)];
        i_curve = i_curve+1;
        
    end
    
    hold off
    set(gca, 'yscale', 'log');
    axis tight
    
    h = legend(hplot,'MCS','CS','location','northeast');
    
    xlabel('$m$','interpreter', 'latex')
    ylabel('Condition number $\mathrm{cond}(A)$','interpreter', 'latex')
    
    set_axis_param
    set_fonts
    
    ax = gca;
    
    if isequal(poly_type,'Hermite')
        if d == 1 || d == 2
            ax.YTick = [1e0 1e5 1e10 1e15];
            ax.YTickLabels = {'10^{0}','10^{5}','10^{10}','10^{15}'};
        elseif d == 4
            ax.YTick = [1e0 1e5 1e10 1e15];
            ax.YTickLabels = {'10^{0}','10^{5}','10^{10}','10^{15}'};
        else
            ax.YTick = [1e0 1e2 1e4 1e6 1e8];
            ax.YTickLabels = {'10^{0}','10^{2}','10^{4}','10^{6}','10^8'};
        end
    else
        if d == 1
            ax.YTick = [1e0 1e5 1e10 1e15];
            ax.YTickLabels = {'10^{0}','10^{5}','10^{10}','10^{15}'};
        elseif d == 2
            ax.YLim = [1,1e4];
            ax.YTick = [1e0 1e2 1e4 1e6];
            ax.YTickLabels = {'10^{0}','10^{2}','10^{4}','10^{6}'};
        else
            ax.YLim = [1,1e2];
            ax.YTick = [1e0 1e1 1e2 1e3];
            ax.YTickLabels = {'10^{0}','10^{1}','10^{2}','10^{3}'};
        end
    end
    
    
    fig_name = ['cond_d',num2str(d),'_N',num2str(N),'_trials',num2str(num_trials)];
    if isequal(poly_type,'Legendre')
        fig_name = ['Legendre_',fig_name];
    else
        fig_name = ['Hermite_',fig_name];
    end
    saveas(fig,['../figs/',fig_name],'epsc');
    
    u = u+1;
    
end

%% plot scalings

% values of p for the hyperbolic cross index set
p_ranges_1 = [20 20 300 ; 7 7 105 ; 3 3 45 ; 1 1 16];
p_ranges_2 = [2 2 18 ; 1 1 12 ; 1 1 11 ; 1 1 9];
p_ranges_3 = [20 20 180 ; 7 7 105 ; 3 3 45 ; 1 1 16];

u = 1;
for d = d_vals
    
    fig = figure(1);
    
    i_curve = 1; hplot = []; % initialize legend function handle
    
    for samp_type = samp_list
        
        if isequal(char(samp_type),'MCS') && isequal(char(poly_type),'Legendre')
            p_ranges = p_ranges_3;
        elseif isequal(char(samp_type),'MCS') && isequal(char(poly_type),'Hermite')
            p_ranges = p_ranges_2;
        else
            p_ranges = p_ranges_1;
        end
        
        % set file name
        file_name = [char(samp_type),'_d',num2str(d),'_N',num2str(N),'_trials',num2str(num_trials)];
        if isequal(poly_type,'Legendre')
            file_name = ['Legendre_scaling_',file_name];
        else
            file_name = ['Hermite_scaling_',file_name];
        end
        
        % load data
        load(['../data/',file_name,'.mat'])
        
        % specify p values
        p_vals = p_ranges(u,1):p_ranges(u,2):p_ranges(u,3);
        
        % compute n values
        n_vals = [];
        for p = p_vals
            n_vals = [n_vals size(generate_index_set('HC',d,p),2)];
        end
        
        hplot = [hplot plot_book_style_each_curve(n_vals,m_vals_data','shaded','mean_std',i_curve)];
        
        i_curve = i_curve+1;
        
    end
    
    m_vals = mean(m_vals_data);
    s = mean(m_vals./(n_vals.*log(n_vals)));
    
    h1 = plot(n_vals , s*n_vals.*log(n_vals),'--','LineWidth',lw,'Color',colors{i_curve});
    h2 = plot(n_vals , n_vals.*log(n_vals)/(d+1),'--','LineWidth',lw,'Color',colors{i_curve+1});
    
    hplot = [hplot h1 h2];
    
    hold off
    axis tight
    
    h = legend(hplot,'MCS','CS','$c n \log(n)$','$ n \log(n) / (d+1)$','location','northwest');
    xlabel('$n$','interpreter', 'latex')
    ylabel('$m$','interpreter', 'latex')
    
    if isequal(poly_type,'Legendre')
        ylim([0 1200]);
    else
        ylim([0 2200]);
    end
    
    set_axis_param
    set_fonts
    
    fig_name = ['scaling_d',num2str(d),'_N',num2str(N),'_trials',num2str(num_trials)];
    if isequal(poly_type,'Legendre')
        fig_name = ['Legendre_',fig_name];
    else
        fig_name = ['Hermite_',fig_name];
    end
    saveas(fig,['../figs/',fig_name],'epsc');
    
    u = u+1;
end
