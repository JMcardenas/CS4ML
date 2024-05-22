%-------------------------------------------------------------------------%
% Filename: figs_App_D_plot.m 
% Author: Juan M. Cardenas 
% Part of the paper "CS4ML: A general sampling framework for Machine 
% Learning".
% Description: generates the plots for Figure 3.
%-------------------------------------------------------------------------%
clear all; close all; clc;
addpath(genpath('../utils'));

% Loop over subfigures % 
for fig_num = 1:3

    if fig_num == 1
        row_init = 1; row_end = 1;
    else
        row_init = 1; row_end = 2;
    end

    for row_num = row_init:row_end
        for col_num = 1:3
    
            fig_name = ['fig_AppD','_', num2str(fig_num),'_',num2str(row_num),'_',num2str(col_num)];
            load(['../../data/Figure App D/',fig_name])
    
            fig = figure();
    
            if col_num == 1
                hPlot = plot_book_style(x_values_data(:,1,1), y_values_data, 'shaded', 'mean_std_log10');
                
                main_figure = 0;
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
            
            saveas(fig,['../../figs/Figure App D/',fig_name],'epsc');
        end
    end
end