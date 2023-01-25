close all;

% close all figures
% LOAD ALEXNET

foldFolder = fullfile('../','FileExchangeEntry/test_75_alex');
fold1_alexnet = fullfile(foldFolder, 'TEST47ALEXNETcustom6_1_among_5_folds.mat');
fold2_alexnet = fullfile(foldFolder, 'TEST47ALEXNETcustom6_2_among_5_folds.mat');
fold3_alexnet = fullfile(foldFolder, 'TEST47ALEXNETcustom6_3_among_5_folds.mat');
fold4_alexnet = fullfile(foldFolder, 'TEST47ALEXNETcustom6_4_among_5_folds.mat');
fold5_alexnet = fullfile(foldFolder, 'TEST47ALEXNETcustom6_5_among_5_folds.mat');

load('-mat',fold1_alexnet);
load('-mat',fold2_alexnet);
load('-mat',fold3_alexnet);
load('-mat',fold4_alexnet);
load('-mat',fold5_alexnet);

% LOAD GOOGLENET

foldFolder2 = fullfile('../','FileExchangeEntry/test_40_googlenet/');
fold1_googlenet = fullfile(foldFolder2, 'GOOGLENET_custom6_1_among_5_folds.mat');
fold2_googlenet = fullfile(foldFolder2, 'GOOGLENET_custom6_2_among_5_folds.mat');
fold3_googlenet = fullfile(foldFolder2, 'GOOGLENET_custom6_3_among_5_folds.mat');
fold4_googlenet = fullfile(foldFolder2, 'GOOGLENET_custom6_4_among_5_folds.mat');
fold5_googlenet = fullfile(foldFolder2, 'GOOGLENET_custom6_5_among_5_folds.mat');

load('-mat',fold1_googlenet);
load('-mat',fold2_googlenet);
load('-mat',fold3_googlenet);
load('-mat',fold4_googlenet);
load('-mat',fold5_googlenet);

% LOAD VGGNET

foldFolder3 = fullfile('../','FileExchangeEntry/test_13_vggnet/');
fold1_vggnet = fullfile(foldFolder3, 'VGGNET_TEST13_1_among_5_folds.mat');
fold2_vggnet = fullfile(foldFolder3, 'VGGNET_TEST13_2_among_5_folds.mat');
fold3_vggnet = fullfile(foldFolder3, 'VGGNET_TEST13_3_among_5_folds.mat');
fold4_vggnet = fullfile(foldFolder3, 'VGGNET_TEST13_4_among_5_folds.mat');
fold5_vggnet = fullfile(foldFolder3, 'VGGNET_TEST13_5_among_5_folds.mat');

load('-mat',fold1_vggnet);
load('-mat',fold2_vggnet);
load('-mat',fold3_vggnet);
load('-mat',fold4_vggnet);
load('-mat',fold5_vggnet);

% LOAD RESNET

foldFolder4 = fullfile('../','FileExchangeEntry/test_9_resnet/');
fold1_resnet = fullfile(foldFolder4, 'RESNET_TEST2_1_among_5_folds.mat');
fold2_resnet = fullfile(foldFolder4, 'RESNET_TEST2_2_among_5_folds.mat');
fold3_resnet = fullfile(foldFolder4, 'RESNET_TEST2_3_among_5_folds.mat');
fold4_resnet = fullfile(foldFolder4, 'RESNET_TEST2_4_among_5_folds.mat');
fold5_resnet = fullfile(foldFolder4, 'RESNET_TEST2_5_among_5_folds.mat');

load('-mat',fold1_resnet);
load('-mat',fold2_resnet);
load('-mat',fold3_resnet);
load('-mat',fold4_resnet);
load('-mat',fold5_resnet);


workspace;  % Make sure the workspace panel is showing.
format long g;
format compact;

predicted_labels_alex = strings([1,452]);
predicted_labels_googlenet = strings([1,452]);
predicted_labels_vggnet = strings([1,452]);
predicted_labels_resnet = strings([1,452]);

scores_aom_alex = [1,452];
scores_csom_alex = [1,452];
scores_earwax_alex = [1,452];
scores_normal_img_alex = [1,452];

scores_aom_googlenet = [1,452];
scores_csom_googlenet = [1,452];
scores_earwax_googlenet = [1,452];
scores_normal_img_googlenet = [1,452];

scores_aom_vggnet = [1,452];
scores_csom_vggnet = [1,452];
scores_earwax_vggnet = [1,452];
scores_normal_vggnet = [1,452];

scores_aom_resnet = [1,452];
scores_csom_resnet = [1,452];
scores_earwax_resnet = [1,452];
scores_normal_resnet = [1,452];

pop = 1;

outputFolder = fullfile('../','/augmented_dataset_histeq_test_extra/dataset');
rootFolder2 = fullfile(outputFolder, 'testing set');
categories = {'aom_train', 'csom_train', 'earwax_train', 'normal_img_train'};
imds = imageDatastore(fullfile(rootFolder2,categories),'LabelSource','foldernames');
actual_labels = imds.Labels;

% Define a starting folder.
start_path = fullfile('../','/augmented_dataset_histeq_test_extra/dataset');
% Ask user to confirm or change.
topLevelFolder = uigetdir(start_path);
% Get list of all subfolders.
allSubFolders = genpath(topLevelFolder);
% Parse into a cell array.
remain = allSubFolders;
listOfFolderNames = {};
while true
	[singleSubFolder, remain] = strtok(remain, ';');
	if isempty(singleSubFolder)
		break;
	end
	listOfFolderNames = [listOfFolderNames singleSubFolder];
end
numberOfFolders = length(listOfFolderNames)

% Process all image files in those folders.
for k = 2 : numberOfFolders
	% Get this folder and print it out.
	thisFolder = listOfFolderNames{k};
	fprintf('Processing folder %s\n', thisFolder);    
	
	% Get PNG files.
	filePattern = sprintf('%s/*.png', thisFolder);
	baseFileNames = dir(filePattern);
	numberOfImageFiles = length(baseFileNames)
	% Now we have a list of all files in this folder.
	
	if numberOfImageFiles >= 1
		% Go through all those image files.
        g = waitbar(0, 'Starting');
		for f = 1: (numberOfImageFiles)

            originalFilename = fullfile(thisFolder, baseFileNames(f).name);
	        originalImage = imread(originalFilename);
            img_alex = augmentedImageDatastore([227 227],originalImage);
            img_googlenet_vgg_resnet = augmentedImageDatastore([224 224],originalImage);
        
            [predicted_item, scores] = classify(netTransfer_alex, img_alex);
            [predicted_item2, scores2] = classify(netTransfer_googlenet, img_googlenet_vgg_resnet);
            [predicted_item3, scores3] = classify(netTransfer_vggnet, img_googlenet_vgg_resnet);
            [predicted_item4, scores4] = classify(netTransfer_resnet, img_googlenet_vgg_resnet);
        
            %PUTS VALUES INTO THE LIST
            predicted_labels_alex(pop) = predicted_item;
            predicted_labels_googlenet(pop) = predicted_item2;
            predicted_labels_vggnet(pop) = predicted_item3;
            predicted_labels_resnet(pop) = predicted_item4;

            scores_aom_alex(pop) = scores(1);
            scores_csom_alex(pop) = scores(2);
            scores_earwax_alex(pop) = scores(3);
            scores_normal_img_alex(pop) = scores(4);

            scores_aom_googlenet(pop) = scores2(1);
            scores_csom_googlenet(pop) = scores2(2);
            scores_earwax_googlenet(pop) = scores2(3);
            scores_normal_img_googlenet(pop) = scores2(4);
            
            scores_aom_vggnet(pop) = scores3(1);
            scores_csom_vggnet(pop) = scores3(2);
            scores_earwax_vggnet(pop) = scores3(3);
            scores_normal_vggnet(pop) = scores3(4);
            
            scores_aom_resnet(pop) = scores4(1);
            scores_csom_resnet(pop) = scores4(2);
            scores_earwax_resnet(pop) = scores4(3);
            scores_normal_resnet(pop) = scores4(4);
        
            pop = pop + 1;


            waitbar(f/numberOfImageFiles, g, sprintf('Folder %d our of %d \n Progress: %d %% ', k-1, numberOfFolders-1, floor(f/numberOfImageFiles*100)));
            pause(0.1);


        end
        
	else
		fprintf('     Folder %s has no image files in it.\n', thisFolder);
    end
end

%Combines all probabilities into 1 single table
alexnet = table(scores_aom_alex', scores_csom_alex', scores_earwax_alex', scores_normal_img_alex');
googlenet = table(scores_aom_googlenet', scores_csom_googlenet', scores_earwax_googlenet', scores_normal_img_googlenet');
vggnet = table(scores_aom_vggnet', scores_csom_vggnet', scores_earwax_vggnet', scores_normal_vggnet');
resnet = table(scores_aom_resnet', scores_csom_resnet', scores_earwax_resnet', scores_normal_resnet');

%Make Column Name for each column
alexnet.Properties.VariableNames = {'aom_train' 'csom_train' 'earwax_train' 'normal_img_train'};
googlenet.Properties.VariableNames = {'aom_train' 'csom_train' 'earwax_train' 'normal_img_train'};
vggnet.Properties.VariableNames = {'aom_train' 'csom_train' 'earwax_train' 'normal_img_train'};
resnet.Properties.VariableNames = {'aom_train' 'csom_train' 'earwax_train' 'normal_img_train'};

%CONVERTS the labels into categorical for EVALUATION
predicted_labels_alex = categorical(predicted_labels_alex);
predicted_labels_googlenet = categorical(predicted_labels_googlenet);
predicted_labels_vggnet = categorical(predicted_labels_vggnet);
predicted_labels_resnet = categorical(predicted_labels_resnet);

%% THIS EVALUATES THE TOP TWO MODELS FOR THE PRESENTED DATA
%ALEXNET MODEL EVALUATION
    %Normal Image Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL=actual_labels;
    PREDICTED=predicted_labels_alex';
    idx = (ACTUAL()=='normal_img_train');
    %disp(idx)
    p = length(ACTUAL(idx));
    n = length(ACTUAL(~idx));
    N = p+n;
    tp = sum(ACTUAL(idx)==PREDICTED(idx));
    tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
    
    %CSOM Validation
    ACTUAL2=actual_labels;
    PREDICTED2= predicted_labels_alex';
    idx2 = (ACTUAL2()=='csom_train');
    %disp(idx)
    p2 = length(ACTUAL2(idx2));
    n2 = length(ACTUAL2(~idx2));
    N2 = p2+n2;
    tp2 = sum(ACTUAL2(idx2)==PREDICTED2(idx2));
    tn2 = sum(ACTUAL2(~idx2)==PREDICTED2(~idx2));
    
    % AOM Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL3=actual_labels;
    PREDICTED3=predicted_labels_alex';
    idx3 = (ACTUAL3()=='aom_train');
    %disp(idx)
    p3 = length(ACTUAL3(idx3));
    n3 = length(ACTUAL3(~idx3));
    N3 = p3+n3;
    tp3 = sum(ACTUAL3(idx3)==PREDICTED3(idx3));
    tn3 = sum(ACTUAL3(~idx3)==PREDICTED3(~idx3));
    
    % Earwax Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL4=actual_labels;
    PREDICTED4=predicted_labels_alex';
    idx4 = (ACTUAL4()=='earwax_train');
    %disp(idx)
    p4 = length(ACTUAL4(idx4));
    n4 = length(ACTUAL4(~idx4));
    N4 = p4+n4;
    tp4 = sum(ACTUAL4(idx4)==PREDICTED4(idx4));
    tn4 = sum(ACTUAL4(~idx4)==PREDICTED4(~idx4));
    
    % Average of All Classification
    N = (N + N2 + N3 + N4)/ 4;
    tp = (tp + tp2 + tp3 + tp4)/4;
    tn = (tn + tn2 + tn3 + tn4)/4;


%GOOGLENET MODEL EVALUATION
    %Normal Image Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL5=actual_labels;
    PREDICTED5=predicted_labels_googlenet';
    idx5 = (ACTUAL5()=='normal_img_train');
    %disp(idx)
    p5 = length(ACTUAL5(idx5));
    n5 = length(ACTUAL5(~idx5));
    N5 = p5+n5;
    tp5 = sum(ACTUAL5(idx5)==PREDICTED5(idx5));
    tn5 = sum(ACTUAL5(~idx5)==PREDICTED5(~idx5));
    
    %CSOM Validation
    ACTUAL6=actual_labels;
    PREDICTED6=predicted_labels_googlenet';
    idx6 = (ACTUAL6()=='csom_train');
    %disp(idx)
    p6 = length(ACTUAL6(idx6));
    n6 = length(ACTUAL6(~idx6));
    N6 = p6+n6;
    tp6 = sum(ACTUAL6(idx6)==PREDICTED6(idx6));
    tn6 = sum(ACTUAL6(~idx6)==PREDICTED6(~idx6));
    
    % AOM Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL7=actual_labels;
    PREDICTED7=predicted_labels_googlenet';
    idx7 = (ACTUAL7()=='aom_train');
    %disp(idx)
    p7 = length(ACTUAL7(idx7));
    n7 = length(ACTUAL7(~idx7));
    N7 = p7+n7;
    tp7 = sum(ACTUAL7(idx7)==PREDICTED7(idx7));
    tn7 = sum(ACTUAL7(~idx7)==PREDICTED7(~idx7));
    
    % Earwax Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL8=actual_labels;
    PREDICTED8=predicted_labels_googlenet';
    idx8 = (ACTUAL8()=='earwax_train');
    %disp(idx)
    p8 = length(ACTUAL8(idx8));
    n8 = length(ACTUAL8(~idx8));
    N8 = p8+n8;
    tp8 = sum(ACTUAL8(idx8)==PREDICTED8(idx8));
    tn8 = sum(ACTUAL8(~idx8)==PREDICTED8(~idx8));
    
    % Average of All Classification
    N5 = (N5 + N6 + N7 + N8)/ 4;
    tp5 = (tp5 + tp6 + tp7 + tp8)/4;
    tn5 = (tn5 + tn6 + tn7 + tn8)/4;

%RESNET MODEL EVALUATION
    %Normal Image Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL9=actual_labels;
    PREDICTED9=predicted_labels_resnet';
    idx9 = (ACTUAL5()=='normal_img_train');
    %disp(idx)
    p9 = length(ACTUAL9(idx9));
    n9 = length(ACTUAL9(~idx9));
    N9 = p9+n9;
    tp9 = sum(ACTUAL9(idx9)==PREDICTED9(idx9));
    tn9 = sum(ACTUAL9(~idx9)==PREDICTED9(~idx9));
    
    %CSOM Validation
    ACTUAL10=actual_labels;
    PREDICTED10=predicted_labels_resnet';
    idx10 = (ACTUAL10()=='csom_train');
    %disp(idx)
    p10 = length(ACTUAL10(idx10));
    n10 = length(ACTUAL10(~idx10));
    N10 = p10+n10;
    tp10 = sum(ACTUAL10(idx10)==PREDICTED10(idx10));
    tn10 = sum(ACTUAL10(~idx10)==PREDICTED10(~idx10));
    
    % AOM Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL11=actual_labels;
    PREDICTED11=predicted_labels_resnet';
    idx11 = (ACTUAL11()=='aom_train');
    %disp(idx)
    p11 = length(ACTUAL11(idx11));
    n11 = length(ACTUAL11(~idx11));
    N11 = p11+n11;
    tp11 = sum(ACTUAL11(idx11)==PREDICTED11(idx11));
    tn11 = sum(ACTUAL11(~idx11)==PREDICTED11(~idx11));
    
    % Earwax Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL12=actual_labels;
    PREDICTED12=predicted_labels_resnet';
    idx12 = (ACTUAL12()=='earwax_train');
    %disp(idx)
    p12 = length(ACTUAL12(idx12));
    n12= length(ACTUAL12(~idx12));
    N12 = p12+n12;
    tp12 = sum(ACTUAL12(idx12)==PREDICTED12(idx12));
    tn12 = sum(ACTUAL12(~idx12)==PREDICTED12(~idx12));
    
    % Average of All Classification
    N9 = (N9 + N10 + N11 + N12)/ 4;
    tp9 = (tp9 + tp10 + tp11 + tp12)/4;
    tn9 = (tn9 + tn10 + tn11 + tn12)/4;


%VGGNET MODEL EVALUATION
    %Normal Image Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL13=actual_labels;
    PREDICTED13=predicted_labels_vggnet';
    idx13 = (ACTUAL13()=='normal_img_train');
    %disp(idx)
    p13 = length(ACTUAL13(idx13));
    n13 = length(ACTUAL13(~idx13));
    N13 = p13+n13;
    tp13 = sum(ACTUAL13(idx13)==PREDICTED13(idx13));
    tn13 = sum(ACTUAL13(~idx13)==PREDICTED13(~idx13));
    
    %CSOM Validation
    ACTUAL14=actual_labels;
    PREDICTED14=predicted_labels_vggnet';
    idx14 = (ACTUAL14()=='csom_train');
    %disp(idx)
    p14 = length(ACTUAL14(idx14));
    n14 = length(ACTUAL14(~idx14));
    N14 = p14+n14;
    tp14 = sum(ACTUAL14(idx14)==PREDICTED14(idx14));
    tn14 = sum(ACTUAL14(~idx14)==PREDICTED14(~idx14));
    
    % AOM Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL15=actual_labels;
    PREDICTED15=predicted_labels_vggnet';
    idx15 = (ACTUAL15()=='aom_train');
    %disp(idx)
    p15 = length(ACTUAL15(idx15));
    n15 = length(ACTUAL15(~idx15));
    N15 = p15+n15;
    tp15 = sum(ACTUAL15(idx15)==PREDICTED15(idx15));
    tn15 = sum(ACTUAL15(~idx15)==PREDICTED15(~idx15));
    
    % Earwax Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL16=actual_labels;
    PREDICTED16=predicted_labels_vggnet';
    idx16 = (ACTUAL16()=='earwax_train');
    %disp(idx)
    p16 = length(ACTUAL16(idx16));
    n16= length(ACTUAL16(~idx16));
    N16 = p16+n16;
    tp16 = sum(ACTUAL16(idx16)==PREDICTED16(idx16));
    tn16 = sum(ACTUAL16(~idx16)==PREDICTED16(~idx16));
    
    % Average of All Classification
  
    N13 = (N13 + N14 + N15 + N16)/ 4;
    tp13 = (tp13 + tp14 + tp15 + tp16)/4;
    tn13 = (tn13 + tn14 + tn15 + tn16)/4;


    %% ACCURACY PER MODEL
    fprintf('\n Accuracy PER MODEL ACCORDING TO CF \n')
    %ALEXNET
    accuracy_alex = (tp+tn)/N

    %GOOGLENET
    accuracy_googlenet = (tp5+tn5)/N5

    %RESNET
    accuracy_resnet = (tp9+tn9)/N9
 
    %VGGNET
    accuracy_vggnet = (tp13+tn13)/N13
    
    
%SORTING TOP TWO PERFORMING MODELS ACCORDING TO CF Accuracies
clear sort;
model_names = {'alexnet'; 'googlenet'; 'vggnet'; 'resnet'};
model_scores = [accuracy_alex; accuracy_googlenet; accuracy_vggnet; accuracy_resnet];
model_predicted_labels = {'predicted_labels_alex';'predicted_labels_googlenet';'predicted_labels_vggnet';'predicted_labels_resnet',};
model_table = table(model_names,model_scores, model_predicted_labels);
%SORTS DESCENDING ORDER
sorted = sortrows(model_table, -2);
%SELECT FIRST HIGH and SECOND HIGH VALUES
max_1 = sorted(1,:);
max_2 = sorted(2,:);
max_3 = sorted(3,:);
max_4 = sorted(4,:);

%Gets the top two models from the table
top1_model = char(max_1.model_names);
top2_model = char(max_2.model_names);
top3_model = char(max_3.model_names);
top4_model = char(max_4.model_names);

%gets the column value in max_1 variable under model_scores column
top1_scores = max_1.model_scores;
top2_scores = max_2.model_scores;


top1_predicted_labels = char(max_1.model_predicted_labels);
%converting string into variable name using function eval
top1_predicted_labels = eval(top1_predicted_labels);
top2_predicted_labels = char(max_2.model_predicted_labels);
top2_predicted_labels = eval(top2_predicted_labels);
top3_predicted_labels = char(max_3.model_predicted_labels);
top3_predicted_labels = eval(top3_predicted_labels);
top4_predicted_labels = char(max_4.model_predicted_labels);
top4_predicted_labels = eval(top4_predicted_labels);

    %% Evaluation of Top 1 Model
    ACTUAL_1=actual_labels;
    PREDICTED_1=top1_predicted_labels';
    idx_1 = (ACTUAL_1()=='normal_img_train');
    %disp(idx)
    p_1 = length(ACTUAL_1(idx_1));
    n_1 = length(ACTUAL_1(~idx_1));
    N_1 = p_1+n_1;
    tp_1 = sum(ACTUAL_1(idx_1)==PREDICTED_1(idx_1));
    tn_1 = sum(ACTUAL_1(~idx_1)==PREDICTED_1(~idx_1));
    fp_1 = n_1-tn_1;
    fn_1 = p_1-tp_1;
    
    tp_rate_1 = tp_1/p_1;
    tn_rate_1 = tn_1/n_1;
    
    %CSOM Validation
    ACTUAL_2=actual_labels;
    PREDICTED_2= top1_predicted_labels';
    idx_2 = (ACTUAL_2()=='csom_train');
    %disp(idx)
    p_2 = length(ACTUAL_2(idx_2));
    n_2 = length(ACTUAL_2(~idx_2));
    N_2 = p_2+n_2;
    tp_2 = sum(ACTUAL_2(idx_2)==PREDICTED_2(idx_2));
    tn_2 = sum(ACTUAL_2(~idx_2)==PREDICTED_2(~idx_2));
    fp_2 = n_2-tn_2;
    fn_2 = p_2-tp_2;
    
    tp_rate_2 = tp_2/p_2;
    tn_rate_2 = tn_2/n_2;
    
    % AOM Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL_3=actual_labels;
    PREDICTED_3=top1_predicted_labels';
    idx_3 = (ACTUAL_3()=='aom_train');
    %disp(idx)
    p_3 = length(ACTUAL_3(idx_3));
    n_3 = length(ACTUAL_3(~idx_3));
    N_3 = p_3+n_3;
    tp_3 = sum(ACTUAL_3(idx_3)==PREDICTED_3(idx_3));
    tn_3 = sum(ACTUAL_3(~idx_3)==PREDICTED_3(~idx_3));
    fp_3 = n_3-tn_3;
    fn_3 = p_3-tp_3;
    
    tp_rate_3 = tp_3/p_3;
    tn_rate_3 = tn_3/n_3;
    
    % Earwax Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL_4=actual_labels;
    PREDICTED_4=top1_predicted_labels';
    idx_4 = (ACTUAL_4()=='earwax_train');
    %disp(idx)
    p_4 = length(ACTUAL_4(idx_4));
    n_4 = length(ACTUAL_4(~idx_4));
    N_4 = p_4+n_4;
    tp_4 = sum(ACTUAL_4(idx_4)==PREDICTED_4(idx_4));
    tn_4 = sum(ACTUAL_4(~idx_4)==PREDICTED_4(~idx_4));
    fp_4 = n_4-tn_4;
    fn_4 = p_4-tp_4;
    
    tp_rate_4 = tp_4/p_4;
    tn_rate_4 = tn_4/n_4;
    
    % Average of All Classification
    
    p_1 = (p_1 + p_2 + p_3 + p_4)/4;
    n_1 = (n_1 + n_2 + n_3 + n_4)/ 4;
    N_1 = (N_1 + N_2 + N_3 + N_4)/ 4;
    tp_1 = (tp_1 + tp_2 + tp_3 + tp_4)/4;
    tn_1 = (tn_1 + tn_2 + tn_3 + tn_4)/4;
    fp_1 = (fp_1 + fp_2 + fp_3 + fp_4)/4;
    fn_1 = (fn_1 + fn_2 + fn_3 + fn_4)/4;
    
    tp_rate_1 = (tp_rate_1 + tp_rate_2 + tp_rate_3 + tp_rate_4)/4;
    tn_rate_1 = (tn_rate_1 + tn_rate_2 + tn_rate_3 + tn_rate_4)/4;

    %% Evaluation of Top 2 Model
    ACTUAL_5=actual_labels;
    PREDICTED_5=top2_predicted_labels';
    idx_5 = (ACTUAL_5()=='normal_img_train');
    %disp(idx)
    p_5 = length(ACTUAL_5(idx_5));
    n_5 = length(ACTUAL_5(~idx_5));
    N_5 = p_5+n_5;
    tp_5 = sum(ACTUAL_5(idx_5)==PREDICTED_5(idx_5));
    tn_5 = sum(ACTUAL_5(~idx_5)==PREDICTED_5(~idx_5));
    fp_5 = n_5-tn_5;
    fn_5 = p_5-tp_5;
    
    tp_rate_5 = tp_5/p_5;
    tn_rate_5 = tn_5/n_5;
    
    %CSOM Validation
    ACTUAL_6=actual_labels;
    PREDICTED_6= top2_predicted_labels';
    idx_6 = (ACTUAL_6()=='csom_train');
    %disp(idx)
    p_6 = length(ACTUAL_6(idx_6));
    n_6 = length(ACTUAL_6(~idx_6));
    N_6 = p_6+n_6;
    tp_6 = sum(ACTUAL_6(idx_6)==PREDICTED_6(idx_6));
    tn_6 = sum(ACTUAL_6(~idx_6)==PREDICTED_6(~idx_6));
    fp_6 = n_6-tn_6;
    fn_6 = p_6-tp_6;
    
    tp_rate_6 = tp_6/p_6;
    tn_rate_6 = tn_6/n_6;
    
    % AOM Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL_7=actual_labels;
    PREDICTED_7=top2_predicted_labels';
    idx_7 = (ACTUAL_7()=='aom_train');
    %disp(idx)
    p_7 = length(ACTUAL_7(idx_7));
    n_7 = length(ACTUAL_7(~idx_7));
    N_7 = p_7+n_7;
    tp_7 = sum(ACTUAL_7(idx_7)==PREDICTED_7(idx_7));
    tn_7 = sum(ACTUAL_7(~idx_7)==PREDICTED_7(~idx_7));
    fp_7 = n_7-tn_7;
    fn_7 = p_7-tp_7;
    
    tp_rate_7 = tp_7/p_7;
    tn_rate_7 = tn_7/n_7;
    
    % Earwax Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL_8=actual_labels;
    PREDICTED_8=top2_predicted_labels';
    idx_8 = (ACTUAL_8()=='earwax_train');
    %disp(idx)
    p_8 = length(ACTUAL_8(idx_8));
    n_8 = length(ACTUAL_8(~idx_8));
    N_8 = p_8+n_8;
    tp_8 = sum(ACTUAL_8(idx_8)==PREDICTED_8(idx_8));
    tn_8 = sum(ACTUAL_8(~idx_8)==PREDICTED_8(~idx_8));
    fp_8 = n_8-tn_8;
    fn_8 = p_8-tp_8;
    
    tp_rate_8 = tp_8/p_8;
    tn_rate_8 = tn_8/n_8;
    
    % Average of All Classification
    
    p_5 = (p_5 + p_6 + p_7 + p_8)/4;
    n_5 = (n_5 + n_6 + n_7 + n_8)/ 4;
    N_5 = (N_5 + N_6 + N_7 + N_8)/ 4;
    tp_5 = (tp_5 + tp_6 + tp_7 + tp_8)/4;
    tn_5 = (tn_5 + tn_6 + tn_7 + tn_8)/4;
    fp_5 = (fp_5 + fp_6 + fp_7 + fp_8)/4;
    fn_5 = (fn_5 + fn_6 + fn_7 + fn_8)/4;
    
    tp_rate_5 = (tp_rate_5 + tp_rate_6 + tp_rate_7 + tp_rate_8)/4;
    tn_rate_5 = (tn_rate_5 + tn_rate_6 + tn_rate_7 + tn_rate_8)/4;

    %% Evaluation of Top 1 Model
    accuracy_1 = (tp_1+tn_1)/N_1;
    sensitivity_1 = tp_rate_1;
    specificity_1 = tn_rate_1;
    precision_1 = tp_1/(tp_1+fp_1);
    recall_1 = sensitivity_1;
    f_measure_1 = 2*((precision_1*recall_1)/(precision_1 + recall_1));
    gmean_1 = sqrt(tp_rate_1*tn_rate_1);
    
    disp(['\n\nTop 1 Model: ' top1_model])
    disp(['accuracy=' num2str(accuracy_1)])
    disp(['sensitivity=' num2str(sensitivity_1)])
    disp(['specificity=' num2str(specificity_1)])
    disp(['precision=' num2str(precision_1)])
    disp(['recall=' num2str(recall_1)])
    disp(['f_measure=' num2str(f_measure_1)])
    disp(['gmean=' num2str(gmean_1)])

    %% Evaluation of Top 2 Model
    accuracy_5 = (tp_5+tn_5)/N_5;
    sensitivity_5 = tp_rate_5;
    specificity_5 = tn_rate_5;
    precision_5 = tp_5/(tp_5+fp_5);
    recall_5 = sensitivity_5;
    f_measure_5 = 2*((precision_5*recall_5)/(precision_5 + recall_5));
    gmean_5 = sqrt(tp_rate_5*tn_rate_5);
    
    disp(['\n\nTop 2 Model: ' top2_model])
    disp(['accuracy=' num2str(accuracy_5)])
    disp(['sensitivity=' num2str(sensitivity_5)])
    disp(['specificity=' num2str(specificity_5)])
    disp(['precision=' num2str(precision_5)])
    disp(['recall=' num2str(recall_5)])
    disp(['f_measure=' num2str(f_measure_5)])
    disp(['gmean=' num2str(gmean_5)])

    figure;
    plotconfusion(actual_labels,top1_predicted_labels')
    clear title;
    title("Confusion Matrix Top 1 Model: " + top1_model);
    
    figure;
    plotconfusion(actual_labels,top2_predicted_labels')
    clear title;
    title("Confusion Matrix Top 2 Model: " + top2_model);

    figure;
    plotconfusion(actual_labels,top3_predicted_labels')
    clear title;
    title("Confusion Matrix Top 3 Model: " + top3_model);

    figure;
    plotconfusion(actual_labels,top4_predicted_labels')
    clear title;
    title("Confusion Matrix Top 4 Model: " + top4_model);

%{    
%TABLE FOR VERIFYING TOP TWO MODELS
clear sort;
model_prob_names = {'alexnet'; 'googlenet'; 'vggnet'; 'resnet'};
temp = [0;0;0;0];
model_prob_table = table(model_prob_names, temp);
sorted_prob = sortrows(model_prob_table, -2);

%FIND top two models names by comparing sorted_prob table (675) and sorted table
%from line 444
for i=1:4
  try
        a = char(sorted.model_names(1));
        aa = char(sorted.model_names(2));
        b = char(sorted_prob.model_prob_names(i));
        if(a == b)
            max_prob_model_1 = char(sorted_prob.model_prob_names(i))
        end  
        if(aa == b)
            max_prob_model_2 = char(sorted_prob.model_prob_names(i))
        end 
  catch
  end
end
%}
    
%%COMBINES ALL MODELS SCORES INTO 1 TABLE FOR ROC
    %Combines all values into 1 more table
    alexnet_scores = table(scores_aom_alex', scores_csom_alex', scores_earwax_alex', scores_normal_img_alex');
    alexnet_scores.Properties.VariableNames = {'aom_train' 'csom_train' 'earwax_train' 'normal_img_train'};

    googlenet_scores = table(scores_aom_googlenet', scores_csom_googlenet', scores_earwax_googlenet', scores_normal_img_googlenet');
    googlenet_scores.Properties.VariableNames = {'aom_train' 'csom_train' 'earwax_train' 'normal_img_train'};

    vggnet_scores = table(scores_aom_vggnet', scores_csom_vggnet', scores_earwax_vggnet', scores_normal_vggnet');
    vggnet_scores.Properties.VariableNames = {'aom_train' 'csom_train' 'earwax_train' 'normal_img_train'};

    resnet_scores = table(scores_aom_resnet', scores_csom_resnet', scores_earwax_resnet', scores_normal_resnet');
    resnet_scores.Properties.VariableNames = {'aom_train' 'csom_train' 'earwax_train' 'normal_img_train'};
    
    %SELECTS TOP 1 and 2 MODEL SCORES FOR ROC CURVE
    top_model_names = {'alexnet'; 'googlenet'; 'vggnet'; 'resnet'};
    top_model_scores = [accuracy_alex; accuracy_googlenet; accuracy_vggnet; accuracy_resnet];
    top_model_predicted_labels = {'alexnet_scores';'googlenet_scores';'vggnet_scores';'resnet_scores',};
    top_model_table = table(top_model_names,top_model_scores, top_model_predicted_labels);
    %SORTS DESCENDING ORDER
    top_sorted = sortrows(top_model_table, -2);
    %SELECT FIRST HIGH and SECOND HIGH VALUES
    top_max_1 = top_sorted(1,:);
    top_max_2 = top_sorted(2,:);
    top_max_3 = top_sorted(3,:);
    top_max_4 = top_sorted(4,:);

    model_1 = char(top_max_1.top_model_names);
    model_2 = char(top_max_2.top_model_names);

    %converting string into variable name using function eval
    top_1_model_scores =char(top_max_1.top_model_predicted_labels);
    top_1_model_scores = eval(top_1_model_scores);
    top_2_model_scores =char(top_max_2.top_model_predicted_labels);
    top_2_model_scores = eval(top_2_model_scores);
    top_3_model_scores =char(top_max_3.top_model_predicted_labels);
    top_3_model_scores = eval(top_3_model_scores);
    top_4_model_scores =char(top_max_4.top_model_predicted_labels);
    top_4_model_scores = eval(top_4_model_scores);
    
    %%ROC CURVE TOP 1 MODEL
    %converts ensemble table into an array
    top_1_roc = table2array(top_1_model_scores);
    top_2_roc = table2array(top_2_model_scores);
    top_3_roc = table2array(top_3_model_scores);
    top_4_roc = table2array(top_4_model_scores);
    %makes the actual labels into double and nominal
    test_labels=double(nominal(imds.Labels));
    
    figure;
    [fp_rate_top1,tp_rate_top1,T_top1,AUC_top1]=perfcurve(test_labels,top_1_roc(:,1),1);
    plot(fp_rate_top1,tp_rate_top1,'LineWidth',1.5)
    hold on
    [fp_rate2_top1,tp_rate2_top1,T2_top1,AUC2_top1]=perfcurve(test_labels,top_1_roc(:,2),2);
    plot(fp_rate2_top1,tp_rate2_top1,'LineWidth',1.5)
    [fp_rate3_top1,tp_rate3_top1,T3_top1,AUC3_top1]=perfcurve(test_labels,top_1_roc(:,3),3);
    plot(fp_rate3_top1,tp_rate3_top1,'LineWidth',1.5)
    [fp_rate4_top1,tp_rate4_top1,T4_top1,AUC4_top1]=perfcurve(test_labels,top_1_roc(:,4),4);
    plot(fp_rate4_top1,tp_rate4_top1,'LineWidth',1.5)
    legend(sprintf('AOM = %f', AUC_top1), sprintf('CSOM = %f', AUC2_top1), sprintf('Earwax = %f', AUC3_top1), sprintf('Normal = %f', AUC4_top1), 'Location','Best')
    %legend('AOM = %d',AUC,'CSOM = %d',AUC2,'Earwax = %d',AUC3, 'Normal Image = %d',AUC4,'Location','Best')
    xlabel('False positive rate'); ylabel('True positive rate');
    clear title;  
    title("Top 1 Model: " + top1_model + " ROC Curve");
    hold off

    %TOP 2 MODEL ROC CURVE
    figure;
    [fp_rate_top2,tp_rate_top2,T_top2,AUC_top2]=perfcurve(test_labels,top_2_roc(:,1),1);
    plot(fp_rate_top2,tp_rate_top2,'LineWidth',1.5)
    hold on
    [fp_rate2_top2,tp_rate2_top2,T2_top2,AUC2_top2]=perfcurve(test_labels,top_2_roc(:,2),2);
    plot(fp_rate2_top2,tp_rate2_top2,'LineWidth',1.5)
    [fp_rate3_top2,tp_rate3_top2,T3_top2,AUC3_top2]=perfcurve(test_labels,top_2_roc(:,3),3);
    plot(fp_rate3_top2,tp_rate3_top2,'LineWidth',1.5)
    [fp_rate4_top2,tp_rate4_top2,T4_top2,AUC4_top2]=perfcurve(test_labels,top_2_roc(:,4),4);
    plot(fp_rate4_top2,tp_rate4_top2,'LineWidth',1.5)
    legend(sprintf('AOM = %f', AUC_top2), sprintf('CSOM = %f', AUC2_top2), sprintf('Earwax = %f', AUC3_top2), sprintf('Normal = %f', AUC4_top2), 'Location','Best')
    %legend('AOM = %d',AUC,'CSOM = %d',AUC2,'Earwax = %d',AUC3, 'Normal Image = %d',AUC4,'Location','Best')
    xlabel('False positive rate'); ylabel('True positive rate');
    clear title;
    title("Top 2 Model: " + top2_model + " ROC Curve");
    hold off

    %TOP 3 MODEL ROC CURVE
    figure;
    [fp_rate_top3,tp_rate_top3,T_top3,AUC_top3]=perfcurve(test_labels,top_3_roc(:,1),1);
    plot(fp_rate_top3,tp_rate_top3,'LineWidth',1.5)
    hold on
    [fp_rate2_top3,tp_rate2_top3,T2_top3,AUC2_top3]=perfcurve(test_labels,top_3_roc(:,2),2);
    plot(fp_rate2_top3,tp_rate2_top3,'LineWidth',1.5)
    [fp_rate3_top3,tp_rate3_top3,T3_top3,AUC3_top3]=perfcurve(test_labels,top_3_roc(:,3),3);
    plot(fp_rate3_top3,tp_rate3_top3,'LineWidth',1.5)
    [fp_rate4_top3,tp_rate4_top3,T4_top3,AUC4_top3]=perfcurve(test_labels,top_3_roc(:,4),4);
    plot(fp_rate4_top3,tp_rate4_top3,'LineWidth',1.5)
    legend(sprintf('AOM = %f', AUC_top3), sprintf('CSOM = %f', AUC2_top3), sprintf('Earwax = %f', AUC3_top3), sprintf('Normal = %f', AUC4_top3), 'Location','Best')
    %legend('AOM = %d',AUC,'CSOM = %d',AUC2,'Earwax = %d',AUC3, 'Normal Image = %d',AUC4,'Location','Best')
    xlabel('False positive rate'); ylabel('True positive rate');
    clear title;
    title("Top 3 Model: " + top3_model + " ROC Curve");
    hold off

    %TOP 4 MODEL ROC CURVE
    figure;
    [fp_rate_top4,tp_rate_top4,T_top4,AUC_top4]=perfcurve(test_labels,top_4_roc(:,1),1);
    plot(fp_rate_top4,tp_rate_top4,'LineWidth',1.5)
    hold on
    [fp_rate2_top4,tp_rate2_top4,T2_top4,AUC2_top4]=perfcurve(test_labels,top_4_roc(:,2),2);
    plot(fp_rate2_top4,tp_rate2_top4,'LineWidth',1.5)
    [fp_rate3_top4,tp_rate3_top4,T3_top4,AUC3_top4]=perfcurve(test_labels,top_4_roc(:,3),3);
    plot(fp_rate3_top4,tp_rate3_top4,'LineWidth',1.5)
    [fp_rate4_top4,tp_rate4_top4,T4_top4,AUC4_top4]=perfcurve(test_labels,top_4_roc(:,4),4);
    plot(fp_rate4_top4,tp_rate4_top4,'LineWidth',1.5)
    legend(sprintf('AOM = %f', AUC_top4), sprintf('CSOM = %f', AUC2_top4), sprintf('Earwax = %f', AUC3_top4), sprintf('Normal = %f', AUC4_top4), 'Location','Best')
    %legend('AOM = %d',AUC,'CSOM = %d',AUC2,'Earwax = %d',AUC3, 'Normal Image = %d',AUC4,'Location','Best')
    xlabel('False positive rate'); ylabel('True positive rate');
    clear title;
    title("Top 4 Model: " + top4_model + " ROC Curve");
    hold off

%Ensemble Classifier will start here
%Creates Array for sotring the values of 1 and 0 per classification
%category
ensemble_aom = [1,452];
ensemble_csom = [1,452];
ensemble_earwax = [1,452];
ensemble_normal_img = [1,452];


for i = 1: length(eval(model_1).aom_train)
    temp = (eval(model_1).aom_train(i) + eval(model_2).aom_train(i))/2;
    %top 1 model's first value under AOM column will be added to top 2
    %model's first value under AOM column divided by 2

    if temp >= 0.5
        ensemble_aom(i) = 1;
    else
        ensemble_aom(i) = 0;
    end

    temp2 = (eval(model_1).csom_train(i) + eval(model_2).csom_train(i))/2;
    if temp2 >= 0.5
        ensemble_csom(i) = 1;
    else
        ensemble_csom(i) = 0;
    end

    temp3 = (eval(model_1).earwax_train(i) + eval(model_2).earwax_train(i))/2;
    if temp3 >= 0.5
        ensemble_earwax(i) = 1;
    else
        ensemble_earwax(i) = 0;
    end

    temp4 = (eval(model_1).normal_img_train(i) + eval(model_2).normal_img_train(i))/2;
    if temp4 >= 0.5
        ensemble_normal_img(i) = 1;
    else
        ensemble_normal_img(i) = 0;
    end
    
    %if ALL values meet 0 thus we will be getting the max values that did
    %not reach 0.5 and make the max value into 1 to prevent null and
    %undefined columns
    if((ensemble_normal_img(i) == 0) && (ensemble_aom(i) == 0) && (ensemble_csom(i) == 0) && (ensemble_earwax(i) == 0))
        A = [temp temp2 temp3 temp4];
        val = max(A);
        
            if val == temp
                ensemble_aom(i) = 1;
            elseif val == temp2
                ensemble_csom(i) = 1;
            elseif val == temp3
                ensemble_earwax(i) = 1;
            elseif val == temp4
                ensemble_normal_img(i) = 1;
            end
        
    end
end

%Combines all values into 1 more table
ensemble_table = table(ensemble_aom', ensemble_csom', ensemble_earwax', ensemble_normal_img');
ensemble_table.Properties.VariableNames = {'aom_train' 'csom_train' 'earwax_train' 'normal_img_train'};

ensemble_predicted_labels = string([1,452]);

%Converts all 1 values into string under specified column name to get the
%predicted values of combined models.
for i = 1:length(eval(model_1).aom_train)
    if ensemble_table.aom_train(i) == 1
        ensemble_predicted_labels(i) = ensemble_table.Properties.VariableNames{1};

    elseif ensemble_table.csom_train(i) == 1 
        ensemble_predicted_labels(i) = ensemble_table.Properties.VariableNames{2};

    elseif ensemble_table.earwax_train(i) == 1
        ensemble_predicted_labels(i) = ensemble_table.Properties.VariableNames{3};

    elseif ensemble_table.normal_img_train(i) == 1
        ensemble_predicted_labels(i) = ensemble_table.Properties.VariableNames{4};
    else
    end
end

%converts to categorial values
ensemble_predicted_labels = categorical(ensemble_predicted_labels);

    figure;
    plotconfusion(actual_labels,ensemble_predicted_labels')
    clear title;
    title("Ensemble Confusion Matrix: " + top1_model + " and " + top2_model);

    %Find out the Ensemble Model Classification Statistics using actual
    %labels and the newly create predicted labels
%% Evaluation Ensemble Model
    ACTUAL_ensemble=actual_labels;
    PREDICTED_ensemble=ensemble_predicted_labels';
    idx_ensemble = (ACTUAL_ensemble()=='normal_img_train');
    %disp(idx)
    p_ensemble = length(ACTUAL_ensemble(idx_ensemble));
    n_ensemble = length(ACTUAL_ensemble(~idx_ensemble));
    N_ensemble = p_ensemble+n_ensemble;
    tp_ensemble = sum(ACTUAL_ensemble(idx_ensemble)==PREDICTED_ensemble(idx_ensemble));
    tn_ensemble = sum(ACTUAL_ensemble(~idx_ensemble)==PREDICTED_ensemble(~idx_ensemble));
    fp_ensemble = n_ensemble-tn_ensemble;
    fn_ensemble = p_ensemble-tp_ensemble;
    
    tp_rate_ensemble = tp_ensemble/p_ensemble;
    tn_rate_ensemble = tn_ensemble/n_ensemble;
    
    %CSOM Validation
    ACTUAL_ensemble2=actual_labels;
    PREDICTED_ensemble2= ensemble_predicted_labels';
    idx_ensemble2 = (ACTUAL_ensemble2()=='csom_train');
    %disp(idx)
    p_ensemble2 = length(ACTUAL_ensemble2(idx_ensemble2));
    n_ensemble2 = length(ACTUAL_ensemble2(~idx_ensemble2));
    N_ensemble2 = p_ensemble2+n_ensemble2;
    tp_ensemble2 = sum(ACTUAL_ensemble2(idx_ensemble2)==PREDICTED_ensemble2(idx_ensemble2));
    tn_ensemble2 = sum(ACTUAL_ensemble2(~idx_ensemble2)==PREDICTED_ensemble2(~idx_ensemble2));
    fp_ensemble2 = n_ensemble2-tn_ensemble2;
    fn_ensemble2 = p_ensemble2-tp_ensemble2;
    
    tp_rate_ensemble2 = tp_ensemble2/p_ensemble2;
    tn_rate_ensemble2 = tn_ensemble2/n_ensemble2;
    
    % AOM Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL_ensemble3=actual_labels;
    PREDICTED_ensemble3=ensemble_predicted_labels';
    idx_ensemble3 = (ACTUAL_ensemble3()=='aom_train');
    %disp(idx)
    p_ensemble3 = length(ACTUAL_ensemble3(idx_ensemble3));
    n_ensemble3 = length(ACTUAL_ensemble3(~idx_ensemble3));
    N_ensemble3 = p_ensemble3+n_ensemble3;
    tp_ensemble3 = sum(ACTUAL_ensemble3(idx_ensemble3)==PREDICTED_ensemble3(idx_ensemble3));
    tn_ensemble3 = sum(ACTUAL_ensemble3(~idx_ensemble3)==PREDICTED_ensemble3(~idx_ensemble3));
    fp_ensemble3 = n_ensemble3-tn_ensemble3;
    fn_ensemble3 = p_ensemble3-tp_ensemble3;
    
    tp_rate_ensemble3 = tp_ensemble3/p_ensemble3;
    tn_rate_ensemble3 = tn_ensemble3/n_ensemble3;
    
    % Earwax Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL_ensemble4=actual_labels;
    PREDICTED_ensemble4=ensemble_predicted_labels';
    idx_ensemble4 = (ACTUAL_ensemble4()=='earwax_train');
    %disp(idx)
    p_ensemble4 = length(ACTUAL_ensemble4(idx_ensemble4));
    n_ensemble4 = length(ACTUAL_ensemble4(~idx_ensemble4));
    N_ensemble4 = p_ensemble4+n_ensemble4;
    tp_ensemble4 = sum(ACTUAL_ensemble4(idx_ensemble4)==PREDICTED_ensemble4(idx_ensemble4));
    tn_ensemble4 = sum(ACTUAL_ensemble4(~idx_ensemble4)==PREDICTED_ensemble4(~idx_ensemble4));
    fp_ensemble4 = n_ensemble4-tn_ensemble4;
    fn_ensemble4 = p_ensemble4-tp_ensemble4;
    
    tp_rate_ensemble4 = tp_ensemble4/p_ensemble4;
    tn_rate_ensemble4 = tn_ensemble4/n_ensemble4;
    
    % Average of All Classification
    
    p_ensemble = (p_ensemble + p_ensemble2 + p_ensemble3 + p_ensemble4)/4;
    n_ensemble = (n_ensemble + n_ensemble2 + n_ensemble3 + n_ensemble4)/ 4;
    N_ensemble = (N_ensemble + N_ensemble2 + N_ensemble3 + N_ensemble4)/ 4;
    tp_ensemble = (tp_ensemble + tp_ensemble2 + tp_ensemble3 + tp_ensemble4)/4;
    tn_ensemble = (tn_ensemble + tn_ensemble2 + tn_ensemble3 + tn_ensemble4)/4;
    fp_ensemble = (fp_ensemble + fp_ensemble2 + fp_ensemble3 + fp_ensemble4)/4;
    fn_ensemble = (fn_ensemble + fn_ensemble2 + fn_ensemble3 + fn_ensemble4)/4;
    
    tp_rate_ensemble = (tp_rate_ensemble + tp_rate_ensemble2 + tp_rate_ensemble3 + tp_rate_ensemble4)/4;
    tn_rate_ensemble = (tn_rate_ensemble + tn_rate_ensemble2 + tn_rate_ensemble3 + tn_rate_ensemble4)/4;

    %% Evaluation of Ensemble Model
    accuracy_ensemble = (tp_ensemble+tn_ensemble)/N_ensemble;
    sensitivity_ensemble = tp_rate_ensemble;
    specificity_ensemble = tn_rate_ensemble;
    precision_ensemble = tp_ensemble/(tp_ensemble+fp_ensemble);
    recall_ensemble = sensitivity_ensemble;
    f_measure_ensemble = 2*((precision_ensemble*recall_ensemble)/(precision_ensemble + recall_ensemble));
    gmean_ensemble = sqrt(tp_rate_ensemble*tn_rate_ensemble);

    
    %%ROC CURVE OF ENSEMBLE CLASSIFIER
    %converts ensemble table into an array
    brandnew = table2array(ensemble_table);
    %makes the actual labels into double and nominal
    test_labels=double(nominal(imds.Labels));
    
    figure;
    [fp_rate,tp_rate,T,AUC]=perfcurve(test_labels,brandnew(:,1),1);
    plot(fp_rate,tp_rate,'LineWidth',1.5)
    hold on
    [fp_rate2,tp_rate2,T2,AUC2]=perfcurve(test_labels,brandnew(:,2),2);
    plot(fp_rate2,tp_rate2,'LineWidth',1.5)
    [fp_rate3,tp_rate3,T3,AUC3]=perfcurve(test_labels,brandnew(:,3),3);
    plot(fp_rate3,tp_rate3,'LineWidth',1.5)
    [fp_rate4,tp_rate4,T4,AUC4]=perfcurve(test_labels,brandnew(:,4),4);
    plot(fp_rate4,tp_rate4,'LineWidth',1.5)
    legend(sprintf('AOM = %f', AUC), sprintf('CSOM = %f', AUC2), sprintf('Earwax = %f', AUC3), sprintf('Normal = %f', AUC4), 'Location','Best')
    %legend('AOM = %d',AUC,'CSOM = %d',AUC2,'Earwax = %d',AUC3, 'Normal Image = %d',AUC4,'Location','Best')
    xlabel('False positive rate'); ylabel('True positive rate');
    clear title;
    title("ROC Curve: Ensemble: " + top1_model + " and " + top2_model);
    hold off
    
    disp(['Ensemble Model'])
    disp(['accuracy=' num2str(accuracy_ensemble)])
    disp(['sensitivity=' num2str(sensitivity_ensemble)])
    disp(['specificity=' num2str(specificity_ensemble)])
    disp(['precision=' num2str(precision_ensemble)])
    disp(['recall=' num2str(recall_ensemble)])
    disp(['f_measure=' num2str(f_measure_ensemble)])
    disp(['gmean=' num2str(gmean_ensemble)])

