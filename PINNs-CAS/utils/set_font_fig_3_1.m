% set font size

font_size = 20;
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

