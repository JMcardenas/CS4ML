%-------------------------------------------------------------------------%
% Filename: figs_3_plot.m 
% Author: Juan M. Cardenas 
% Part of the paper "CS4ML: A general sampling framework for Machine 
% Learning".
% Description: generates the plots for Figure 3.
%-------------------------------------------------------------------------%
clear all; close all; clc;
addpath(genpath('../utils'));

% Loop over subfigures %
fig_num = 3;

for row_num = 1
    for col_num = 1:3

        fig_name = ['fig','_',num2str(fig_num),'_',num2str(row_num),'_',num2str(col_num)];
        load(['../../data/Figure 3/',fig_name])

        fig = figure();

        if col_num == 1
            hPlot = plot_book_style(x_values_data(:,1,1), y_values_data, 'shaded', 'mean_std_log10');
            
            main_figure = 1;
            set_legend
            beautify_plot 
        
        elseif (col_num == 2) || (col_num == 3)
            
            dPlot = densityScatterChart(x_values_data(:,1),y_values_data(:,1), 'UseColor', true, 'UseAlpha', true);
    
            dPlot.DensityMethod = 'ksdensity';
            dPlot.AlphaRange    = [.2 .5];

            set_font_fig_3_1
        
        else
            disp('incorrect column number: try 1 or 2');
        end
        
        saveas(fig,['../../figs/Figure 3/',fig_name],'epsc');
    end
end
