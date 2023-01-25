
outputFolder = fullfile('../','tympanic_membrane_dataset');
rootFolder = fullfile(outputFolder, 'dataset');
categories = {'earwax', 'csom', 'aom', 'normal_img'};


%% Load Training Images
% In order for imageDataStore to parse the folder names as category labels,
% you would have to store image categories in corresponding sub-folders.
allImages = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');

% Balance the Datasets 
tbl = countEachLabel(allImages);
minSetCount = min(tbl{:,2})

allImages = splitEachLabel(allImages,minSetCount, 'randomized');
countEachLabel(allImages);

%% Split data into training and test sets 
[trainingImages, testImages] = splitEachLabel(allImages, 0.7, 'randomize');
 
%% Load Pre-trained Network (AlexNet) 
alex = alexnet; 

%% Review Network Architecture 
layers = alex.Layers 

%% Modify Pre-trained Network 
% AlexNet was trained to recognize 1000 classes, we need to modify it to
% recognize just 4 classes. 
layers(23) = fullyConnectedLayer(4); % change this based on # of classes
layers(25) = classificationLayer

%% Perform Transfer Learning
opts = trainingOptions( ...
    'sgdm', ...
    'InitialLearnRate', 0.0001,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',16, ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 64, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'auto');

%% Set custom read function
trainingImages.ReadFcn = @readFunctionTrain;

%% Train the Network 
myNet = trainNetwork(trainingImages, layers, opts);

%% Test Network Performance
% Now let's the test the performance of our new "snack recognizer" on the test set.
testImages.ReadFcn = @readFunctionTrain;
predictedLabels = classify(myNet, testImages); 
accuracy = mean(predictedLabels == testImages.Labels)

