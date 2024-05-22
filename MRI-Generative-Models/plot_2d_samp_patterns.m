load('~/scratch/braingen_GCS_example/test_run_trials_image_data_128/K_tilde_lines.mat')
b = max(K_tilde_2d(:))
a = min(K_tilde_2d(:))
imshow(imresize(reshape(K_tilde_2d,[128,128]),6))
colorbar
set(gca,'ColorScale','log')
set(gca,'CLim',[a,b])
set(gcf,'Color',[1.00,1.00,1.00])