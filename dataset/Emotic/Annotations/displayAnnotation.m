function [ ] = displayAnnotation(I, body_bbox, anns_con, anns_cat, personID, Npeople, gender, age )
% displayAnnotation(I, body_bbox, annscon, anncats )
% I = original size image 
% body_bbox = body bounding box
% anncats --> structure holding categories' for the image.
% annscon --> structure holding valence, arousal and dominance values. 

% subplot(m,n,...)
m = 5;
n = 11;

x = abs(floor(body_bbox(1)));
y = abs(floor(body_bbox(2)));
w = abs(floor(body_bbox(3)) - ...
    floor(body_bbox(1)));
h = abs(floor(body_bbox(4)) - ...
    floor(body_bbox(2)));

% main image with the body bounding box
figure('units','normalized','outerposition',[0 0 1 1])
subplot(m,n,1:44), imshow(I), hold on,
title(sprintf('Annotated person: %d/%d; %s; %s',personID,Npeople,gender,age), 'FontSize', 20, 'FontWeight', 'bold')
rectangle('Position', [x y w h],'Edgecolor','r','Linewidth',5);

% plotting categorical annotations
%subplot(3,2,5), axis off
if length(anns_cat) > 3
    l = ceil(length(anns_cat)/2);
    y = linspace(0,1,l);
    y = fliplr(y);
    ind = 1;
    z = 1;
    while ind < (length(anns_cat)+1)
        subplot(m,n,46:49),axis off
        text(0, y(z), anns_cat{ind}, 'FontSize', 20, 'FontWeight', 'bold')
        ind = ind +1;
        if ind <= length(anns_cat)
            text(0.5, y(z), anns_cat{ind}, 'FontSize', 20, 'FontWeight', 'bold')
            ind = ind +1;
        end
        z = z +1;
    end
else
    y = [1 0.6 0.2];
    subplot(m,n,46:49),axis off
    for x = 1:length(anns_cat)
        text(0, y(x), anns_cat{x}, 'FontSize', 20, 'FontWeight', 'bold')
    end
end

% plotting continous annotations
subplot(m,n,51:54) 
bar(([anns_con.valence anns_con.arousal anns_con.dominance])), hold on
xlim = get(gca, 'xlim');
plot(xlim,[5 5], '--k')
axis([-inf,inf,0, 10])
set(gca,'XTick', 1:3, 'XTickLabel',{'V', 'A', 'D'}, 'FontSize', 20, 'FontWeight', 'bold')
hold off

end