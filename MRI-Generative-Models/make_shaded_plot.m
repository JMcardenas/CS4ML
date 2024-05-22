% size(datamatrix)

mean_data = squeeze(mean(datamatrix,2));

% size(mean_data)

slice_data = squeeze(mean(datamatrix(:,:,:,30:90),4));

% size(slice_data)

% should be (num_m, num_trials, num_plots)

hPlot = plot_book_style(1:num_percs, slice_data(:,:,jmin+1), 'shaded', plot_type, 1, '-');
hold on
% hPlotRaw = plot(1:16, slice_data(:,:,1),'*b');
hold on
hPlot = plot_book_style(1:num_percs, slice_data(:,:,jmin), 'shaded', plot_type, 2, '-');
hold on
% hPlotRaw = plot(1:16, slice_data(:,:,2),'*r');
beautify_plot
xlabel('sampling percentage','Interpreter','LaTeX');
ylabel('average PSNR (dB) on frames 30--90','Interpreter','LaTeX');
samp_percs = ["0.00125", "0.0025", "0.00375", "0.005", "0.00625", "0.0075", "0.00875", "0.01", "0.01125", "0.0125", "0.01375", "0.015", "0.01625", "0.0175", "0.01875", "0.02", "0.03", "0.04", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3"];
% xticks([1,3,5,7,9,11,13,15])
xticks([2,4,6,8,10,12,14,16])
%xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
% xticklabels(["0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875", "1.0", "1.125", "1.25", "1.375", "1.5", "1.625", "1.75", "1.875", "2.0"])
% xticklabels(["0.125", "0.375", "0.625", "0.875", "1.125", "1.375", "1.625", "1.875"])
xticklabels(["0.25", "0.5", "0.75", "1.0",  "1.25", "1.5", "1.75", "2.0"])
set(gca,'xlim',[1,16]);
legend('','','','MCS','','','','CS')
set(gca,'yscale','linear')