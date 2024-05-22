%--- Description ---%
%
% Filename: shaded_plot.m
% Authors: Ben Adcock, Simone Brugiapaglia and Clayton Webster
% Part of the book "Sparse polynomial approximation of high-dimensional functions", SIAM
%
% Description: Given random data associated 
% The plot shows the mean curve and a filled-in area defined by [mean - std
% dev, mean + std dev]. % For each curve, the data is assumed to be 
% relative to the same set of x values and to be generated using a fixed 
% number of random trials for each (x,y) value.
% 
% Input: 
% x_data - x-values (they should be the same for all curves)
% y_data - 3D array containing the y-values defiing the curves. This array
% should be structured as follows: y_data(i, j, k) contains the y-value 
% corresponding to the i-th x-value (i.e., x(i)), to the j-th random trial, 
% and the k-th curve.
% stats - specifies what statistics of the data should be visualized.
%         The available options are:
%         'mean_std_log10': 10^( mean(log10(data)) +/- std(log10(data)) )
%         'mean_std': mean +/- std dev
%         'mean_std_eps': max(mean +/- std dev, machine precision)
%         'mean_sem': mean +/- standard error of the mean (SEM) 
%
% Output:
% hMeanPlots - handle correspoing to the mean plots only (to be used to add
% a legend)

function [hMeanPlots] = shaded_plot_comparison(x_data, y_data, stats)

x_data = x_data(:)'; % make sure x is a column vector

% Extract y_data dimensions
n_x     = size(y_data,1);
n_trial = size(y_data,2);
n_curve = size(y_data,3);

% get default plotting parameters
[ms, lw, fs, colors, markers] = get_fig_param();

% change colors and markers for comparison plot
for i = 1:length(colors)
    new_colors{2*i-1} = colors{i};
    new_colors{2*i}   = 0.8 * colors{i};
end
new_markers = markers;
for i = 2:2:length(markers)
    new_markers{i} = ['-',markers{i}];
end
colors = new_colors;
markers = new_markers;


hMeanPlots = []; % initialize legend function handle
for i_curve = 1 : n_curve
    
    % Extract data realtive to current curve
    y_data_curve = y_data(:,:,i_curve);
    
    % Define curves bounding the shaded region
    
    
    switch stats
        case 'mean_std_log10'
            % compute statistics on log10 of data, i.e.
            % 10^( mean(log10(data)) +/- std(log10(data)) )
            mean_curve = 10.^(mean(log10(y_data_curve),2))';
            std_curve = std(log10(y_data_curve),0,2)';
            curve_min = 10.^(log10(mean_curve) - std_curve);
            curve_max = 10.^(log10(mean_curve) + std_curve);
        case 'mean_std'
            % mean +/- std dev
            mean_curve = mean(y_data_curve,2)';
            std_curve = std(y_data_curve,0,2)';
            curve_min = mean_curve - std_curve;
            curve_max = mean_curve + std_curve;
        case 'mean_std_eps'
            % max(mean +/- std dev, machine precision)
            mean_curve = mean(y_data_curve,2)';
            std_curve = std(y_data_curve,0,2)';
            curve_min = max(mean_curve - std_curve, eps);
            curve_max = mean_curve + std_curve;
        case 'mean_sem'
            % mean +/- standard error of the mean (SEM) 
            mean_curve = mean(y_data_curve,2)';
            std_curve = std(y_data_curve,0,2)'/sqrt(n_trial);
            curve_min = mean_curve - std_curve;
            curve_max = mean_curve + std_curve;
        otherwise
            error('not implemented')
    end
    
    % Plot shaded region boundary
    AlphaLevel = 0.075;
    
    plot(x_data, curve_min, 'color', [colors{i_curve}, AlphaLevel]);
    hold on
    plot(x_data, curve_max, 'color', [colors{i_curve}, AlphaLevel])
    
    % Fill in shaded region 
    x_data_2 = [x_data, fliplr(x_data)];
    inBetween = [curve_min, fliplr(curve_max)];
    fill(x_data_2, inBetween, colors{i_curve}, 'FaceAlpha', AlphaLevel, 'EdgeAlpha', 0);
    
    % Plot mean curve
    h = plot(x_data, mean_curve, markers{i_curve},...
        'markersize',ms,'MarkerFaceColor',colors{i_curve},...
        'MarkerEdgeColor',colors{i_curve},'LineWidth',lw,...
        'Color',colors{i_curve});
    hMeanPlots = [hMeanPlots, h]; % add current mean plot to legend handle
end
hold off


