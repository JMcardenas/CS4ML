%--- Description ---%
%
% Filename: plot_mean_quantiles.m
% Authors: Ben Adcock, Simone Brugiapaglia and Clayton Webster
% Part of the book "Sparse polynomial approximation of high-dimensional functions", SIAM
%
% Description: Given random data associated 
% The plot shows the mean curve and a filled-in area between two quantiles
% For each curve, the data is assumed to be relative to the same set of x 
% values and to be generated using a fixed number of random trials for each
% (x,y) value.
% 
% Input: 
% x - x-values used for all curves
%
% y_data - 3D array containing the y-values defiing the curves. This array
% should be structured as follows: y_data(i, j, k) contains the y-value 
% corresponding to the i-th x-value (i.e., x(i)), to the j-th random trial, 
% and the k-th curve.
%
% prob (optional) - vector with two components, corresponding to the 
% probabilities defining the quantiles and, in turn, the shaded areas. The
% default value is [0.1, 0.9].
%
% Output:
% hMeanPlots - handle correspoing to the mean plots only (to be used to add
% a legend)

function [hMeanPlots] = plot_mean_quantiles(x, y_data, prob)

% Check if the vector prob is assigned as input or not
if nargin < 3
    prob =[0.1, 0.9]; % assign default value
end
prob = sort(prob, 'ascend'); 

% Extract y_data dimensions
n_x     = size(y_data,1);
n_curve = size(y_data,3);

% get default plotting parameters
[ms, lw, fs, colors, markers] = get_fig_param();

hMeanPlots = []; % initialize legend function handle
for i_curve = 1 : n_curve
    
    % Extract data realtive to current curve
    y_data_curve = y_data(:,:,i_curve); 
        
    % Define quantile curves
    quantile_min = zeros(1, n_x);
    quantile_max = zeros(1, n_x);
    for i_x = 1 : n_x
        quantile_min(i_x) = quantile(y_data_curve(i_x,:), prob(1));
        quantile_max(i_x) = quantile(y_data_curve(i_x,:), prob(2));
    end
    plot(x, quantile_min, 'color', colors{i_curve});
    hold on
    plot(x, quantile_max, 'color', colors{i_curve})
    
    % Shaded region
    x2 = [x, fliplr(x)];
    inBetween = [quantile_min, fliplr(quantile_max)];
    fill(x2, inBetween, colors{i_curve}, 'FaceAlpha', 0.2, 'EdgeAlpha', 0);
    
    % Mean curve
    h = plot(x, mean(y_data_curve,2)', markers{i_curve},...
        'markersize',ms,'MarkerFaceColor',colors{i_curve},...
        'MarkerEdgeColor',colors{i_curve},'LineWidth',lw,...
        'Color',colors{i_curve});
    hMeanPlots = [hMeanPlots, h]; % add current mean plot to legend handle
end
hold off


