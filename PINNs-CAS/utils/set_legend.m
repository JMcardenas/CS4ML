%-------------------------------------------------------------------------%
% Filename: set_legend.m 
% Authors: Juan M. Cardenas .
% Part of the paper "CS4ML: A general sampling framework for Machine 
% Learning".
%
% Description: generates legends for Figures 3.
%-------------------------------------------------------------------------%
% Inputs:
% fig_num - figure number (3)
% row_num - row number (1)
% col_num - column number (either 1 or 2)
%-------------------------------------------------------------------------%

space = ' ';

if main_figure == 1
    name_curve_1 = ['$\tanh$',space,'$5\times50$','-', 'CAS'];  
    name_curve_2 = ['$\tanh$',space,'$5\times50$','-', 'MC'];  
    name_curve_3 = ['sigmoid',space,'$5\times50$','-', 'CAS'];  
    name_curve_4 = ['sigmoid',space,'$5\times50$','-', 'MC'];  

elseif fig_num == 1
    name_curve_1 = ['ReLU',space,'$5\times50$','-', 'CAS'];  
    name_curve_2 = ['ReLU',space,'$5\times50$','-', 'MC'];  
    name_curve_3 = ['ELU',space,'$5\times50$','-', 'CAS'];  
    name_curve_4 = ['ELU',space,'$5\times50$','-', 'MC'];  

elseif fig_num == 2
    if row_num == 1
        name_curve_1 = ['$\tanh$',space,'$3\times30$','-', 'CAS'];  
        name_curve_2 = ['$\tanh$',space,'$3\times30$','-', 'MC'];  
        name_curve_3 = ['sigmoid',space,'$3\times30$','-', 'CAS'];  
        name_curve_4 = ['sigmoid',space,'$3\times30$','-', 'MC'];  
    
    elseif row_num == 2
        name_curve_1 = ['ReLU',space,'$3\times30$','-', 'CAS'];  
        name_curve_2 = ['ReLU',space,'$3\times30$','-', 'MC'];  
        name_curve_3 = ['ELU',space,'$3\times30$','-', 'CAS'];  
        name_curve_4 = ['ELU',space,'$3\times30$','-', 'MC'];  
    else
        disp('incorrect row number try 1 or 2');
    end

elseif fig_num == 3
    if row_num == 1
        name_curve_1 = ['$\tanh$',space,'$10\times100$','-', 'CAS'];  
        name_curve_2 = ['$\tanh$',space,'$10\times100$','-', 'MC'];  
        name_curve_3 = ['sigmoid',space,'$10\times100$','-', 'CAS'];  
        name_curve_4 = ['sigmoid',space,'$10\times100$','-', 'MC'];  
    
    elseif row_num == 2
        name_curve_1 = ['ReLU',space,'$10\times100$','-', 'CAS'];  
        name_curve_2 = ['ReLU',space,'$10\times100$','-', 'MC'];  
        name_curve_3 = ['ELU',space,'$10\times100$','-', 'CAS'];  
        name_curve_4 = ['ELU',space,'$10\times100$','-', 'MC'];  
    else
        disp('incorrect row number try 1 or 2');
    end

else
    disp('incorrect fig_num: try 1, 2, or 3. Or main_figure = 1');
end

leg = legend(hPlot, name_curve_1, name_curve_2, name_curve_3, name_curve_4);

location_plot = 'best';
num_cols = 1;
font_size = 20;

set(leg,'Interpreter','LaTex',...
        'Location',location_plot,...
        'color','white','box','on',...
        'fontsize',font_size,...
        'NumColumns',num_cols,...
        'Orientation','Horizontal');
