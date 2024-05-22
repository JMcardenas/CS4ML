clear all; close all; clc
samp_percs = ["0.00125", "0.0025", "0.00375", "0.005", "0.00625", "0.0075", "0.00875", "0.01", "0.01125", "0.0125", "0.01375", "0.015", "0.01625", "0.0175", "0.01875", "0.02"]; %, "0.03", "0.04", "0.05", "0.1", "0.15", "0.2", "0.25", "0.3"];
num_trials = 20;
N = 128;
plot_type = 'mean_std';

num_percs = size(samp_percs,2)

datamatrix = zeros(num_percs,20,2,N);
all_psnr_avgs = zeros(num_percs,2,N);

jmin = 1;

for i = 1:num_percs
    for j = jmin:jmin+1
        psnr_trials = zeros(1,N);
        for k = 1:num_trials
            samp_perc = convertStringsToChars(samp_percs(i));
            file_str = ['~/scratch/braingen_GCS_example/test_run_trials_image_data_128/' ...
                        'DNN_run_data_' samp_perc '_method_' num2str(j) '_trial_' num2str(k) '.mat'];
            load(file_str)
%             psnr_values

            datamatrix(i,k,j,:) = psnr_values;
            psnr_trials = psnr_trials + psnr_values;
        end
        psnr_trials = psnr_trials/num_trials;
        all_psnr_avgs(i,j,:) = psnr_trials;
    end 
end

% return

% colors = [0    0.4470    0.7410
%     0.8500    0.3250    0.0980
%     0.9290    0.6940    0.1250
%     0.4940    0.1840    0.5560
%     0.4660    0.6740    0.1880
%     0.6350    0.0780    0.1840
%     0.3010    0.7450    0.9330
%     1.0000    1.0000    0.0700
%     0.0700    0.6200    1.0000
%     1.0000    0.4100    0.1600
%     0.3900    0.8300    0.0700
%     0.7200    0.2700    1.0000
%     0.0600    1.0000    1.0000
%     1.0000    0.0700    0.6500
%     1.0000    0.0000    0.0000
%     1.0000    0.0000    1.0000];
colors = [0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.4940    0.1840    0.5560
    0.9290    0.6940    0.1250
    0.4660    0.6740    0.1880
    0.6350    0.0780    0.1840
    0.3010    0.7450    0.9330
    1.0000    1.0000    0.0700
    0.0700    0.6200    1.0000
    1.0000    0.4100    0.1600
    0.3900    0.8300    0.0700
    0.7200    0.2700    1.0000
    0.0600    1.0000    1.0000
    1.0000    0.0700    0.6500
    1.0000    0.0000    0.0000
    1.0000    0.0000    1.0000];

samp_percs(1)
samp_percs(3)
samp_percs(5)

jjj = 1;
lll = '-';
for i = 1:2:5%num_percs-2
    samp_perc = convertStringsToChars(samp_percs(i));
    hPlot = plot_book_style(1:N, squeeze(datamatrix(i,:,jmin,:))', 'shaded', plot_type,jjj,lll);
    hold on
    jjj = jjj + jjj;
end

beautify_plot

jjj = 1;
lll = '-.';
for i = 1:2:5%num_percs-2
    samp_perc = convertStringsToChars(samp_percs(i));
    hPlot = plot_book_style(1:N, squeeze(datamatrix(i,:,jmin+1,:))', 'shaded', plot_type,jjj,lll);
    hold on
    jjj = jjj + jjj;
end

beautify_plot
set(gca,'xlim',[30,90]);
set(gca,'ylim',[5,40]);
set(gca,'yscale','linear')
samp_percentages = ["0.125", "0.25", "0.375", "0.5", "0.625", "0.75", "0.875", "1.0", "1.125", "1.25", "1.375", "1.5", "1.625", "1.75", "1.875", "2.0"];
legend('','','',samp_percentages(1) + '\% CS','','','',samp_percentages(3) + '\% CS','','','',samp_percentages(5)+ '\% CS','','','',samp_percentages(1) + '\% MCS','','','',samp_percentages(3) + '\% MCS','','','',samp_percentages(5)+'\% MCS')
