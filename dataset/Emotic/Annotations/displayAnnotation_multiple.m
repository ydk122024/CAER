function [ ] = displayAnnotation_multiple(I, person)

% font size for the display
sizeCats = 16;

% subplot(m,n,...)
n = max(length(person.annotations_categories),length(person.annotations_continuous)) + 1;
m = 2;
% n = number of annotators + 1
% n = length(person.annotations_categories) + length(person.annotations_continuous) + 1
% m = 2;

% extract bounding box coordinates
x = abs(floor(person.body_bbox(1)));
y = abs(floor(person.body_bbox(2)));
w = abs(floor(person.body_bbox(3)) - ...
    floor(person.body_bbox(1)));
h = abs(floor(person.body_bbox(4)) - ...
    floor(person.body_bbox(2)));

% display image with the body bounding box
figure('units','normalized','outerposition',[0 0 1 1])
subplot(m,n,1), imshow(I), hold on,
rectangle('Position', [x y w h],'Edgecolor','r','Linewidth',5);
drawnow
title(sprintf('%s , %s',person.age, person.gender),'FontSize', 15);

y = [0.9:-0.08:-0.9];

% print discrete annotation per each annotator
ndisc = length(person.annotations_categories);
for a = 1:min(length(person.annotations_categories),6)
    
    subplot(m,n,a + 1),axis off
    title(sprintf('Annotator: %d', a), 'FontSize', 20, 'FontWeight', 'bold')
    
    % plotting categorical annotations
    for c = 1:length(person.annotations_categories(a).categories)
        subplot(m,n,a + 1),axis off 
        % text(1, c + 1, person.annotations_categories(a).categories{c}, 'FontSize', sizeCats, 'FontWeight', 'bold')
        text(0.2, y(c), person.annotations_categories(a).categories{c}, 'FontSize', sizeCats, 'FontWeight', 'bold')
        drawnow
    end
end

% print continuous annotation per each annotator
for a = 1:min(length(person.annotations_continuous),6);
    subplot(m,n,n + 1 + a),axis off
    title(sprintf('Annotator: %d', ndisc + a), 'FontSize', 20, 'FontWeight', 'bold')
    % plotting continuous
    text(0.2, y(1), sprintf('Valence: %d / 10',person.annotations_continuous(a).valence), 'FontSize', sizeCats, 'FontWeight', 'bold')
    text(0.2, y(2), sprintf('Arousal: %d / 10',person.annotations_continuous(a).arousal), 'FontSize', sizeCats, 'FontWeight', 'bold')
    text(0.2, y(3), sprintf('Dominance: %d / 10',person.annotations_continuous(a).dominance), 'FontSize', sizeCats, 'FontWeight', 'bold')
end

