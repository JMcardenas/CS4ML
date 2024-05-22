%--- Description ---%
%
% Filename: set_fonts.m
% Authors: Ben Adcock, Simone Brugiapaglia and Clayton Webster
% Part of the book "Sparse Polynomial Approximation of High-Dimensional
% Functions", SIAM, 2021
%
% Description: loads a set of graphical parameters related to fonts
%
% Update (May 2023): modified by the authors of "CS4ML: A general framework
% for active learning with arbitrary data based on Christoffel functions"
%
% Description: removed the x and y label routines

[ms, lw, fs, colors, markers] = get_fig_param();

set(gca, 'FontSize', fs)

hLegend = findobj(gcf, 'Type', 'Legend');
set(hLegend, 'interpreter', 'latex', 'fontsize',fs)
