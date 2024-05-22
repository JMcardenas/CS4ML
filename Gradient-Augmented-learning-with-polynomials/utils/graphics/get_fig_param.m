%--- Description ---%
%
% Filename: get_fig_param.m
% Authors: Ben Adcock, Simone Brugiapaglia and Clayton Webster
% Part of the book "Sparse Polynomial Approximation of High-Dimensional
% Functions", SIAM, 2021
%
% Description: loads a set of default plotting parameters
%
% Outputs:
% ms - markersize
% lw - linewidth
% fs - fontsize
% colors - RGB colors
% markers - plot markers
% AlphaLevel - degree of transparency in shaded plots
%
% Update (May 2023): modified by the authors of "CS4ML: A general framework
% for active learning with arbitrary data based on Christoffel functions"
%
% Description: changed the values of ms, lw and fs for better visualization

function [ms, lw, fs, colors, markers, AlphaLevel] = get_fig_param()

ms = 10;
lw = 2;
fs = 24;

colors = {[0    0.4470    0.7410],...
    [0.8500    0.3250    0.0980],...
    [0.9290    0.6940    0.1250],...
    [0.4940    0.1840    0.5560],...
    [0.4660    0.6740    0.1880],...
    [0.3010    0.7450    0.9330],...
    [0.6350    0.0780    0.1840]} ;

markers = {'-*','-o','-s','-^','-v','-+','-x'};

AlphaLevel = 0.1;