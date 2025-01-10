%% This code loads the Annotations and displays them randomly
%  Set the following paths: paths.Annotations, paths.images, select 

clearvars
close all
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
        Annotations = train;
    case 'validation'
        Annotations = val;
    case 'test'
        Annotations = test;
end

% show images along with all the annotations (train data split has juts one annotator,
% the other dataset splits have 2-6 annotators per image in the testing
% set, 2-10 annotators per image in the validation set.

ind = 1:length(Annotations); % display sorted images
% ind = randperm(length(Annotations)); % display images randomly sorted
for i = ind
        
    % read image
    I = imread(fullfile(paths.images,Annotations(i).folder, Annotations(i).filename));
    
    % show annotations
    for p = 1:length(Annotations(i).person)
                
        displayAnnotation_multiple(I, Annotations(i).person(p));
        pause;
        close
    end
    
end
    
       