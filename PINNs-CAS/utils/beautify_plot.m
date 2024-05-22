% Parameters to improve the plot

font_size = 19;
gcf;
set(gcf,'Color',[1 1 1]);

set(findall(gcf,'type','text'),'fontSize',font_size);
set(findall(gcf,'type','axes'),'fontsize',font_size);
set(gcf,'DefaultTextFontSize',font_size);

% Position plot at left hand corner with width 10 and height 10. 
set(gcf, 'PaperPosition', [0 0 10 10]); 

%Set the paper to have width 10 and height 10. 
set(gcf, 'PaperSize', [10 10]);
set(gcf,'Position',[1 1 800 800]);

ax          = gca;
ax.Position = [0.16 0.14 0.80 0.79];
nplots      = size(ax.Children,1);
 
% Set limits
xmin = 1e16;    xmax = 1e-16;
ymin = 1e16;    ymax = 1e-16;

for j = 1:nplots
     
    ax.Children(j).LineWidth = 1.5;

    if min(ax.Children(j).XData) < xmin
        xmin = min(ax.Children(j).XData);
    end
    if max(ax.Children(j).XData) > xmax
        xmax = max(ax.Children(j).XData);
    end
    if min(ax.Children(j).YData) < ymin
        ymin = min(ax.Children(j).YData);
    end
    if max(ax.Children(j).YData) > ymax
        ymax = max(ax.Children(j).YData);
    end
end

if xmin < 0 
    xmin = xmin*1.1;
else
    xmin = xmin*0.9; 
end

if ymin < 0 
    ymin = ymin*1.1;
else
    ymin = ymin*0.9; 
end

ymax = ymax*5.5;   
xmax = xmax; 
 
set(gca,'xlim',[xmin,xmax]);
set(gca,'ylim',[ymin,ymax]);
set(gca,'LineWidth',1.5)

% add grid and box
grid on;
box on;  

% set yscale
set(gca, 'yscale', 'log')
axis tight

% set labels 
xlabel('$m$ ','Interpreter','LaTex','fontsize',font_size+20);
ylabel('test error $E(u)$','Interpreter','LaTeX','fontsize',font_size+20);
 