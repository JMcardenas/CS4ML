load('~/scratch/braingen_GCS_example/test_run_trials_image_data_128/K_tilde.mat')
viewer = viewer3d(BackgroundColor="white", ...
    GradientColor=[0.5 0.5 0.5]*2,Lighting="on")
viewer.CameraPosition = [213 113 63];
viewer.CameraTarget = [64.5 64.5 64.5];
viewer.CameraUpVector = [0.8568 0.1751 0.4849];
viewer.CameraZoom = 5.5;
drawnow

intensity = [0 20 40 120 220 1024];
alpha = [0.00 0.20 0.4 0.6 0.8 1.0];
color = [0 0 0; 43 10 10; 103 37 20; 199 155 97; 216 213 201; 255 255 255]/255;
queryPoints = linspace(min(intensity),max(intensity),256);
alphamap = interp1(intensity,alpha,queryPoints)';
colormap = interp1(intensity,color,queryPoints);

% tform = affinetform3d(A);

hFig = viewer.Parent;
filename = "animation.gif";

for i = 1:101
%     clear(viewer)
    plot_data = reshape(K_tilde_iterations(i,:),[128,128,128]);
    if i == 1
        volshow(plot_data, ...
            Parent=viewer,Colormap=colormap,Alphamap=alphamap);%,Transformation=tform);
    else
        viewer.Children(1).Data = plot_data;
    end
%     pause
    I = getframe(hFig);
    [indI,cm] = rgb2ind(I.cdata,256);
    if i == 1
        imwrite(indI,cm,filename,"gif",Loopcount=inf,DelayTime=0)
    else
        imwrite(indI,cm,filename,"gif",WriteMode="append",DelayTime=0)
    end
%     pause
end