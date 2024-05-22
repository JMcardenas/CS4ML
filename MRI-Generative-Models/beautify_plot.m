
gcf;

set(gcf,'Color',[1 1 1]);

font_size = 20;

set(findall(gcf,'type','text'),'fontSize',font_size);
set(findall(gcf,'type','axes'),'fontsize',font_size);
set(gcf,'DefaultTextFontSize',font_size);

%Position plot at left hand corner with width 10 and height 10.
set(gcf, 'PaperPosition', [0 0 10 10]); 
%Set the paper to have width 10 and height 10.
set(gcf, 'PaperSize', [10 10]);

ax = gca;

nplots = size(ax.Children,1);

xlabel('image number','Interpreter','LaTeX');
ylabel('PSNR (dB)','Interpreter','LaTeX');

for j = 1:nplots
    ax.Children(j).LineWidth = 1.5;
end

set(gca,'LineWidth',1.0)

grid on;
box on;

leg = legend;
set(leg,'Interpreter','LaTex',...
      'Location','best',...
      'color','white','box','on',...
      'fontsize',18);
