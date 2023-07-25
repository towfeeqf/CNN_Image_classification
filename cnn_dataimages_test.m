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
imagesTrainPath28 = fullfile(workingDir,"train28/");

%%
if ~exist(imagesTrainPath28,'dir')
    mkdir(imagesTrainPath28);
    report = batchmyimagesresizefcn(imagesTrainPath, imagesTrainPath28);
else
    fprintf('I am here');
end
%%

imdsTrainSet = imageDatastore(imagesTrainPath28, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
%[imdsTrain, imdsValid]=splitEachLabel(imdsTrainSet,0.8,'randomize');

tr_numClasses = numel(unique(imdsTrainSet.Labels));
tbl = countEachLabel(imdsTrainSet);

%
% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2});
%
% Limit the number of images to reduce the time it takes
% run this example.
maxNumImages = 100;
minSetCount = min(maxNumImages,minSetCount);

% % Use splitEachLabel method to trim the set.
[imdsTrainSet2] = splitEachLabel(imdsTrainSet, minSetCount, 'randomize');
 
% % Notice that each set now has exactly the same number of images.
 countEachLabel(imdsTrainSet2);

 [imdsTrain, imdsValid]=splitEachLabel(imdsTrainSet2,0.8,'randomize');

 img = readimage(imdsTrain,1);
 inputSize = size(img);
 numClasses = numel(unique(imdsTrain.Labels));



layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
   
%    dropoutLayer;

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer  

   maxPooling2dLayer(2,'Stride',2)
%   dropoutLayer;

%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer   
%     
%      maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,32,'Padding','same')
%     batchNormalizationLayer
%     reluLayer   
 
 
    %%%%
 
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];






%  layers = [
%     imageInputLayer(inputSize)
%     convolution2dLayer(5,20)
%     batchNormalizationLayer
%     reluLayer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];



options = trainingOptions('sgdm', ...
    'MiniBatchSize',20,...
    'MaxEpochs',4, ...
    'ValidationData',imdsValid, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);

% probs probabilities
[YPred, probs] = classify(net,imdsValid);
YValid = imdsValid.Labels;
accuracy = mean(YPred == YValid)


figure
set(gcf,'Position',[360 100 600 520]);

%cm = confusionchart(YPred,YValid);
cm1 = confusionchart(YValid,YPred);

%%
%%
idx = randperm(numel(imdsValid.Files),16);
figure

for i = 1:16
    subplot(4,4,i)
    
    
    correctLabel = imdsValid.Labels(idx(i));
    
    
    I = readimage(imdsValid,idx(i));
    imshow(I)
    predLabel = YPred(idx(i));
    
    if predLabel == correctLabel
        
        color = "\color{blue}";
    else
        color = "\color{red}";
    end
    
   predClassTitle = string(predLabel);
   percent_str = num2str(100*max(probs(idx(i),:)),3)+ "%";
   
   %predClassTitle = strrep(string(predLabel),'_',' ');
    %correctClassTitle = strrep(string(correctClass),'_',' ');
    
%    
%     t =title( "Predicted: " + color + predClassTitle  + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
% 
%     set(t,'Interpreter','none');

    title( color + predClassTitle  + ", " + percent_str);

    %title( color+predClassTitle  + ", " + num2str(100*max(probs(idx(i),:)),3) + "%",'Interpreter','none');

    %set(0,'DefaultTextInterpreter','none')

end

sgtitle('Random Samples Testing','FontSize',12,'FontWeight','bold');

%%
imagesTestPath = fullfile(workingDir,"test/");
imagesTestPath28 = fullfile(workingDir,"test28/");


if ~exist(imagesTestPath28,'dir')
    mkdir(imagesTestPath28);
    reporttest = batchmyimagesresizefcn(imagesTestPath, imagesTestPath28);
else
    fprintf('I am in testing');
end

%
imdsTestSet = imageDatastore(imagesTestPath28, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);


%
% probs probabilities
[YPred_test, probs_test] = classify(net,imdsTestSet)
%%
heatmap(probs_test)


%%

%  %% resize all the images into smaller size to save memory and time of execution
% 
% train_outputPath = fullfile(workingDir,"train_output/");
% %result2 = batchmyimagesresizefcn(imagesTrainPath, train_outputPath);
% 
% 
%  %%
%  
%  %imagesTestPath = fullfile(workingDir,"test/");
% 
% imdsTrainSet28 = imageDatastore(train_outputPath, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
% [imdsTrain28, imdsValid28]=splitEachLabel(imdsTrainSet28,0.8,'randomize');
% 
% %imdsTest =  imageDatastore(imagesTestPath, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
% 
% 
% 
% %% to know the size of one file in imds data store
% img28 = readimage(imdsTrainSet28,1);
% inputSize28 = size(img28);
% numClasses28 = numel(unique(imdsTrainSet28.Labels));
% 
% 
% layers = [
%     imageInputLayer(inputSize28)
%     convolution2dLayer(5,20)
%     batchNormalizationLayer
%     reluLayer
%     fullyConnectedLayer(numClasses28)
%     softmaxLayer
%     classificationLayer];
% 
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',20,...
%     'MaxEpochs',4, ...
%     'ValidationData',imdsValid, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% net = trainNetwork(imdsTrain,layers,options);
% 
% YPred = classify(net,imdsValid);
% YValid = imdsValid.Labels;
% accuracy = mean(YPred == YValid)
% 
% 
% figure
% set(gcf,'Position',[360 100 600 520]);
% 
% %cm = confusionchart(YPred,YValid);
% cm1 = confusionchart(YValid,YPred);
% 
% % %%
% % layers = [
% %     imageInputLayer(inputSize)
% %     
% %     convolution2dLayer(3,8,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer   
% %     
% %     maxPooling2dLayer(2,'Stride',2)
% %     
% %     convolution2dLayer(3,16,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer   
% %     
% %     maxPooling2dLayer(2,'Stride',2)
% %     
% %     convolution2dLayer(3,32,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer   
% %     maxPooling2dLayer(2,'Stride',2)
% %    
% % %    dropoutLayer;
% % 
% %     convolution2dLayer(3,64,'Padding','same')
% %     batchNormalizationLayer
% %     reluLayer  
% % 
% %    maxPooling2dLayer(2,'Stride',2)
% % %   dropoutLayer;
% % 
% % %     convolution2dLayer(3,32,'Padding','same')
% % %     batchNormalizationLayer
% % %     reluLayer   
% % %     
% % %      maxPooling2dLayer(2,'Stride',2)
% % %     
% % %     convolution2dLayer(3,32,'Padding','same')
% % %     batchNormalizationLayer
% % %     reluLayer   
% %  
% %  
% %     %%%%
% %  
% %     fullyConnectedLayer(numClasses)
% %     softmaxLayer
% %     classificationLayer];
% 
% %%
% options = trainingOptions('sgdm', ...
%     'MiniBatchSize',20,...
%     'MaxEpochs',4, ...
%     'ValidationData',imdsValid28, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% net28 = trainNetwork(imdsTrain28,layers,options);
% 
% YPred28 = classify(net28,imdsValid28);
% YValid28 = imdsValid28.Labels;
% accuracy = mean(YPred == YValid28)
% 
% %%
% figure
% set(gcf,'Position',[360 100 600 520]);
% 
% %cm = confusionchart(YPred,YValid);
% cm1 = confusionchart(YValid28,YPred);
