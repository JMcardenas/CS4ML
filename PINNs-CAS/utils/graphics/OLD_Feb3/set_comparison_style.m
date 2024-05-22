% Changes line style and colors for plots where 
function set_comparison_style(hPlot)

[ms, lw, fs, colors, markers] = get_fig_param();

for i = 1:length(hPlot)
    if mod(i,2) == 0
        set(hPlot(i), 'LineStyle', '--')
    end
    set(hPlot(i), 'Color', colors{ceil(i/2)})
    set(hPlot(i), 'MarkerFaceColor', colors{ceil(i/2)})
    set(hPlot(i), 'MarkerEdgeColor', colors{ceil(i/2)})
end