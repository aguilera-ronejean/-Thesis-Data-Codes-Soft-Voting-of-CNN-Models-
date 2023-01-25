
    outputFolder = fullfile('../','tympanic_membrane_dataset/');
rootFolder = fullfile(outputFolder, 'dataset');
categories = {'aom', 'csom', 'earwax', 'normal_img'};


%% Load Training Images
% In order for imageDataStore to parse the folder names as category labels,
% you would have to store image categories in corresponding sub-folders.
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');

% Balance the Datasets 
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2}) %undersampling
imds = splitEachLabel(imds,minSetCount, 'randomized');
countEachLabel(imds)

%% Split data into training and test sets 
[trainingImages, testImages] = splitEachLabel(imds, 0.5, 'randomize');
trainingImages = augmentedImageDatastore([227 227],trainingImages);
testImages = augmentedImageDatastore([227 227],testImages);

%% Load Pre-trained Network (AlexNet) 
alex = alexnet; 


%% Modify Pre-trained Network 
% AlexNet was trained to recognize 1000 classes, we need to modify it to
% recognize just 4 classes. 
% alexNet Architecture 
net=alexnet;
 % Replacing the last layers with new layers
layersTransfer = net.Layers(1:end-7);%81.1 7 %-3 75%

% Number of categories
numClasses = 4; %set number of categories
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer 
    classificationLayer];

%% Perform Transfer Learning
options = trainingOptions( ...
        'sgdm',...
        'ExecutionEnvironment','auto',...
        'MaxEpochs',1, ...
        'MiniBatchSize',64,...
        'Shuffle','every-epoch', ...
        'L2Regularization', 0.0001, ...
        'InitialLearnRate',0.0001, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',16, ...
        'Verbose',false, ...
        'Plots','training-progress');

%% Train the Network 
netTransfer = trainNetwork(trainingImages, layers, options);

%% Test Network Performance
% Now let's the test the performance of our new "snack recognizer" on the test set.
analyzeNetwork(netTransfer)
actual_labels = imds.Labels;
[predicted_labels, posterior] = classify(netTransfer, testImages);
% Save the Independent ResNet Architectures obtained for each Fold
    save(sprintf('ALEXNET_SAVED'),'netTransfer', 'predicted_labels', "posterior", 'imds', 'actual_labels');

% Confusion Matrix
figure;
plotconfusion(actual_labels,predicted_labels')
title('Confusion Matrix: alexnet');

%ROC CURVE
test_labels=double(nominal(imds.Labels));

% ROC Curve
[fp_rate,tp_rate,T,AUC]=perfcurve(test_labels,posterior);
figure;
plot(fp_rate,tp_rate,'b-');
grid on;
xlabel('False Positive Rate');
ylabel('Detection Rate');
% Area under the ROC curve value
AUC()

%Normal Image Evaluation
%Evaluate(YValidation,YPred)
ACTUAL=actual_labels;
PREDICTED=predicted_labels';
idx = (ACTUAL()=='normal_img');
%disp(idx)
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;

tp_rate = tp/p;
tn_rate = tn/n;

%CSOM Validation
ACTUAL2=actual_labels;
PREDICTED2=predicted_labels';
idx2 = (ACTUAL2()=='csom');
%disp(idx)
p2 = length(ACTUAL2(idx2));
n2 = length(ACTUAL2(~idx2));
N2 = p2+n2;
tp2 = sum(ACTUAL2(idx2)==PREDICTED2(idx2));
tn2 = sum(ACTUAL2(~idx2)==PREDICTED2(~idx2));
fp2 = n2-tn2;
fn2 = p2-tp2;

tp_rate2 = tp2/p2;
tn_rate2 = tn2/n2;

% AOM Evaluation
%Evaluate(YValidation,YPred)
ACTUAL3=actual_labels;
PREDICTED3=predicted_labels';
idx3 = (ACTUAL3()=='aom');
%disp(idx)
p3 = length(ACTUAL3(idx3));
n3 = length(ACTUAL3(~idx3));
N3 = p3+n3;
tp3 = sum(ACTUAL3(idx3)==PREDICTED3(idx3));
tn3 = sum(ACTUAL3(~idx3)==PREDICTED3(~idx3));
fp3 = n3-tn3;
fn3 = p3-tp3;

tp_rate3 = tp3/p3;
tn_rate3 = tn3/n3;

% Earwax Evaluation
%Evaluate(YValidation,YPred)
ACTUAL4=actual_labels;
PREDICTED4=predicted_labels';
idx4 = (ACTUAL4()=='earwax');
%disp(idx)
p4 = length(ACTUAL4(idx4));
n4 = length(ACTUAL4(~idx4));
N4 = p4+n4;
tp4 = sum(ACTUAL4(idx4)==PREDICTED4(idx4));
tn4 = sum(ACTUAL4(~idx4)==PREDICTED4(~idx4));
fp4 = n4-tn4;
fn4 = p4-tp4;

tp_rate4 = tp4/p4;
tn_rate4 = tn4/n4;

% Average of All Classification

p = (p + p2 + p3 + p4)/4;
n = (n + n2 + n3 + n4)/ 4;
N = (N + N2 + N3 + N4)/ 4;
tp = (tp + tp2 + tp3 + tp4)/4;
tn = (tn + tn2 + tn3 + tn4)/4;
fp = (fp + fp2 + fp3 + fp4)/4;
fn = (fn + fn2 + fn3 + fn4)/4;

tp_rate = (tp_rate + tp_rate2 + tp_rate3 + tp_rate4)/4;
tn_rate = (tn_rate + tn_rate2 + tn_rate3 + tn_rate4)/4;

% Evaluation
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
AUC_OVERALL = (AUC + AUC2 + AUC3 + AUC4)/4;

disp(['accuracy=' num2str(accuracy)])
disp(['sensitivity=' num2str(sensitivity)])
disp(['specificity=' num2str(specificity)])
disp(['precision=' num2str(precision)])
disp(['recall=' num2str(recall)])
disp(['f_measure=' num2str(f_measure)])
disp(['gmean=' num2str(gmean)])
title={'''AUC_OVERALL','''AUC_AOM','''AUC_CSOM','''AUC_EARWAX','''AUC_NORMAL','''accuracy','''sensitivity','''specificity','''precision','''recall','''f_measure','''gmean'};
VALUES=[AUC_OVERALL,AUC,AUC2,AUC3,AUC4,accuracy,sensitivity,specificity,precision,recall,f_measure,gmean];
filename='performance.xlsx';
xlswrite(filename,title,'Sheet1','A18'); %Always change to empty cell
xlswrite(filename,VALUES,'Sheet1', 'A19'); %Always change to empty cell
 
%accuracy = mean(predicted_labels == actual_labels)

