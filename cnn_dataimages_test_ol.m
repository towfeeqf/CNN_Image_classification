%tic; % Start timer.
clc; % Clear command window.
close all;

all_fig = findall(0, 'type', 'figure');
close(all_fig);

imtool close all;
clearvars; % Get rid of variables from prior run of this m-file.
fprintf('Running the program...\n'); % Message sent to command window.
workspace; % Make sure the workspace panel with all the variables is showing.
imtool close all;  % Close all imtool figures.
%format long g;
format compact;
format bank
%captionFontSize = 10;
%toc; % end timer

%%

workingDir = pwd;
imagesTrainPath = fullfile(workingDir,"train/");
imagesTestPath = fullfile(workingDir,"test/");

imdsTrainSet = imageDatastore(imagesTrainPath, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
%[imdsTrain, imdsValid]=splitEachLabel(imdsTrainSet,0.8,'randomize');

imdsTest =  imageDatastore(imagesTestPath, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
%%
tr_numClasses = numel(unique(imdsTrainSet.Labels));
tbl = countEachLabel(imdsTrainSet);

%tbl = countEachLabel(imdsTest) % this is not valid because, test folder
%has no categories in it.

%%
% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2}) ;

% Limit the number of images to reduce the time it takes
% run this example.
maxNumImages = 100;
minSetCount = min(maxNumImages,minSetCount);

% % Use splitEachLabel method to trim the set.
[imdsTrainSet2, imdsValidSet2] = splitEachLabel(imdsTrainSet, minSetCount, 'randomize');
 
% % Notice that each set now has exactly the same number of images.
 countEachLabel(imdsTrainSet2);

%% to know the size of one file in imds data store
img1 = readimage(imdsTrainSet2,1);



inputSize = size(img1)
numClasses = numel(unique(imdsTrainSet2.Labels));


layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(5,20)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'MiniBatchSize',20,...
    'MaxEpochs',4, ...
    'ValidationData',imdsValidSet2, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrainSet2,layers,options);

YPred = classify(net,imdsValidSet2);
YValidation = imdsValidSet2.Labels;
accuracy = mean(YPred == YValidSet2)



