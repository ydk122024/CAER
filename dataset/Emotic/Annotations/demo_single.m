%% This code loads the Annotations and displays them
%  Set the following paths: paths.Annotations, paths.images, select

clearvars
clc


% **********************************************************************

% specify the data split you want to visualize: train, validation or test
% datasplit = 'train';
datasplit = 'validation';
% datasplit = 'test';

% folder containing Annotations.mat
paths.Annotations = '../Database/emotic_annotations';

% folder containing emotic images into subfolders 'mscoco', 'ade20k', 'framesdb', 'emodb_small'
paths.images = '../Database/emotic';

% **********************************************************************

% load the Annotations
load(fullfile(paths.Annotations, 'Annotations.mat'))

% select the data split
switch datasplit
    case 'train'
        select = 'train';
    case 'validation'
        select = 'val';
    case 'test'
        select = 'test';
end


% NOTE: train, test, and val have almost the same structure. The only difference
% is that val contains up to 10 annotations per sample (each sample has been
% annotated by up to 10 different annotators), test contains up to 6 annotations per sample (each sample has been
% annotated by up to 6 different annotators), and train contains just 2
% annotators per image (one for continuous dimensions and one for
% categories)
eval(['Annotations = ' select])


%% loops over the annotation, showing one for each worker and each person in the image
for indimg = 1:length(Annotations)
    
    % read the image
    I = imread(fullfile(paths.images,Annotations(indimg).folder, Annotations(indimg).filename));
    
    % go through all the annotated people in that image
    Npeople = length(Annotations(indimg).person);
    for indper = 1:Npeople
        
        % read body bounding box coordinates
        body_bbox = Annotations(indimg).person(indper).body_bbox;
        
        % select one annotation index randomly
        indAnn_cat = randperm(length(Annotations(indimg).person(indper).annotations_categories));
        indAnn_cat = indAnn_cat(1);
        indAnn_cont = randperm(length(Annotations(indimg).person(indper).annotations_continuous));
        indAnn_cont = indAnn_cont(1);
        
        % read the selected annotations and show image along with the
        % annotations
        anns_con = Annotations(indimg).person(indper).annotations_continuous(indAnn_cont);
        anns_cat = Annotations(indimg).person(indper).annotations_categories(indAnn_cat).categories;
        gender = Annotations(indimg).person(indper).gender;
        age = Annotations(indimg).person(indper).age;
        displayAnnotation(I,body_bbox, anns_con, anns_cat, indper, Npeople, gender, age)
        pause
        close all
        
    end
end
        

