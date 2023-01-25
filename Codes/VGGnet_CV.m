tic
    outputFolder = fullfile('../','/augmented_dataset_undersampled_histeq');
    rootFolder = fullfile(outputFolder, 'dataset/training set');
    categories = {'aom', 'csom', 'earwax', 'normal'};
    imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
    
    % Balance the Datasets 
    
    tbl = countEachLabel(imds)
    %minSetCount = min(tbl{:,2}) % Undersampling to get Balanced Data
    %imds = splitEachLabel(imds,minSetCount, 'randomized');
    
    % Determine the split up
    %total_split=countEachLabel(imds)

    % Number of Images
    num_images=length(imds.Labels);
    
    %%K-fold Validation-
    % Number of folds
    num_folds=5;
    
    % Loop for each fold
    for fold_idx=1:num_folds
        
        fprintf('Processing %d among %d folds \n',fold_idx,num_folds);
        
       % Test Indices for current fold
        vgg_test_idx=fold_idx:num_folds:num_images;
    
        % Test cases for current fold
         imdsTest = subset(imds,vgg_test_idx);
         vgg_labeltest=countEachLabel(imdsTest)
        % Train indices for current fold
        vgg_train_idx=setdiff(1:length(imds.Files),vgg_test_idx);
        
        % Train cases for current fold
        imdsTrain = subset(imds,vgg_train_idx);
    
        imdsValidation = subset(augmentedImageDatastore([224 224],imds),vgg_test_idx);
        
        vgg_labeltrain= countEachLabel(imdsTrain)
    
        % googlenet Architecture 
        net = vgg16;
        layersTransfer = net.Layers(1:end-3);
        
        save net;
        % Number of categories
        numClasses = 4; %set number of categories
    
        %Repalce Layers
        layers = [
                    layersTransfer
                    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
                    softmaxLayer
                    classificationLayer
                ];
    
       
        %rmsprop,sgdm,adam
        
        % Training Options, we choose a small mini-batch size due to limited images 
        options = trainingOptions( ...
            'sgdm', ...
            'MaxEpochs',5, ...
            'MiniBatchSize',24,...
            'Shuffle','every-epoch', ...
            'ValidationData',imdsValidation, ...
            'LearnRateSchedule','piecewise', ...
            'ValidationFrequency',60, ...
            'Verbose', true, ...
            'L2Regularization', 0.0001, ...
            'InitialLearnRate',0.0002, ...
            'OutputNetwork', 'best-validation-loss', ...
            'Plots','training-progress');
    
       % Resizing all training images to [224 224] for Googlenet architecture
        auimds = augmentedImageDatastore([224 224],imdsTrain);
        
        
        % Training
        netTransfer_vggnet = trainNetwork(auimds,layers,options);
        
        % Resizing all testing images to [224 224] for Googlenet architecture   
         augtestimds = augmentedImageDatastore([224 224],imdsTest);
       
        % Testing and their corresponding Labels and vgg_posterior for each Case
        [vgg_predicted_labels(vgg_test_idx),vgg_posterior(vgg_test_idx,:)] = classify(netTransfer_vggnet, augtestimds)
        % Save the Independent ResNet Architectures obtained for each Fold
        save(sprintf('VGGNET_TEST15_%d_among_%d_folds',fold_idx,num_folds),'netTransfer_vggnet','vgg_test_idx','vgg_train_idx','vgg_labeltest','vgg_labeltrain', 'vgg_predicted_labels', "vgg_posterior");
        
        % Clearing unnecessary variables 
        clearvars -except fold_idx num_folds num_images vgg_predicted_labels vgg_posterior imds netTransfer_vggnet net numClasses;
        
    end
    allImages = augmentedImageDatastore([224 224],imds);
    analyzeNetwork(netTransfer_vggnet);
    %%Performance Study
    % Actual Labels
    actual_labels=imds.Labels;
    
    % Confusion Matrix
    figure;
    plotconfusion(actual_labels,vgg_predicted_labels')
    clear title;
    title('Confusion Matrix: VGGNet');
    
    %ROC CURVE
    test_labels=double(nominal(imds.Labels));
    
    % ROC Curves of ALL Combined into 1 Plot
    figure;
    [fp_rate,tp_rate,T,AUC]=perfcurve(test_labels,vgg_posterior(:,1),1);
    plot(fp_rate,tp_rate,'LineWidth',1.5)
    hold on
    [fp_rate2,tp_rate2,T2,AUC2]=perfcurve(test_labels,vgg_posterior(:,2),2);
    plot(fp_rate2,tp_rate2,'LineWidth',1.5)
    [fp_rate3,tp_rate3,T3,AUC3]=perfcurve(test_labels,vgg_posterior(:,3),3);
    plot(fp_rate3,tp_rate3,'LineWidth',1.5)
    [fp_rate4,tp_rate4,T4,AUC4]=perfcurve(test_labels,vgg_posterior(:,4),4);
    plot(fp_rate4,tp_rate4,'LineWidth',1.5)
    legend(sprintf('AOM = %f', AUC), sprintf('CSOM = %f', AUC2), sprintf('Earwax = %f', AUC3), sprintf('Normal = %f', AUC4), 'Location','Best')
    %legend('AOM = %d',AUC,'CSOM = %d',AUC2,'Earwax = %d',AUC3, 'Normal Image = %d',AUC4,'Location','Best')
    xlabel('False positive rate'); ylabel('True positive rate');
    clear title;
    title('ROC Curves of Classificiations')
    hold off
       
    
    
    %Normal Image Evaluation
    %Evaluate(YValidation,YPred)
    ACTUAL=actual_labels;
    PREDICTED=vgg_predicted_labels';
    idx = (ACTUAL()=='normal');
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
    PREDICTED2=vgg_predicted_labels';
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
    PREDICTED3=vgg_predicted_labels';
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
    PREDICTED4=vgg_predicted_labels';
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
    
    %Softmax probability for each class
    ACTUAL_label=actual_labels;
    PREDICTED_normal=vgg_predicted_labels';
    target = (PREDICTED_normal()=='normal');
    class_1 = (ACTUAL_label()=='normal');
    class_2 = (ACTUAL_label()=='aom');
    class_3 = (ACTUAL_label()=='csom');
    class_4 = (ACTUAL_label()=='earwax');
    
    %Normal Softmax
    prob_1_normal = (sum(ACTUAL_label(class_1)==PREDICTED_normal(class_1))/length(ACTUAL_label(class_1))*100);
    prob_2_normal = (sum(ACTUAL_label(class_2)==PREDICTED_normal(class_1))/length(ACTUAL_label(class_1))*100);
    prob_3_normal = (sum(ACTUAL_label(class_3)==PREDICTED_normal(class_1))/length(ACTUAL_label(class_1))*100);
    prob_4_normal = (sum(ACTUAL_label(class_4)==PREDICTED_normal(class_1))/length(ACTUAL_label(class_1))*100);
    
    disp('Normal Softmax')
    disp(['normal=' num2str(prob_1_normal)])
    disp(['aom=' num2str(prob_2_normal)])
    disp(['csom=' num2str(prob_3_normal)])
    disp(['earwax=' num2str(prob_4_normal)])
    
    %AOM Softmax
    prob_1_aom = (sum(ACTUAL_label(class_1)==PREDICTED_normal(class_2))/length(ACTUAL_label(class_2))*100);
    prob_2_aom = (sum(ACTUAL_label(class_2)==PREDICTED_normal(class_2))/length(ACTUAL_label(class_2))*100);
    prob_3_aom = (sum(ACTUAL_label(class_3)==PREDICTED_normal(class_2))/length(ACTUAL_label(class_2))*100);
    prob_4_aom = (sum(ACTUAL_label(class_4)==PREDICTED_normal(class_2))/length(ACTUAL_label(class_2))*100);
    
    disp('AOM Softmax')
    disp(['normal=' num2str(prob_1_aom)])
    disp(['aom=' num2str(prob_2_aom)])
    disp(['csom=' num2str(prob_3_aom)])
    disp(['earwax=' num2str(prob_4_aom)])
    
    %CSOM Softmax
    prob_1_csom = (sum(ACTUAL_label(class_1)==PREDICTED_normal(class_3))/length(ACTUAL_label(class_3))*100);
    prob_2_csom = (sum(ACTUAL_label(class_2)==PREDICTED_normal(class_3))/length(ACTUAL_label(class_3))*100);
    prob_3_csom = (sum(ACTUAL_label(class_3)==PREDICTED_normal(class_3))/length(ACTUAL_label(class_3))*100);
    prob_4_csom = (sum(ACTUAL_label(class_4)==PREDICTED_normal(class_3))/length(ACTUAL_label(class_3))*100);
    
    disp('CSOM Softmax')
    disp(['normal=' num2str(prob_1_csom)])
    disp(['aom=' num2str(prob_2_csom)])
    disp(['csom=' num2str(prob_3_csom)])
    disp(['earwax=' num2str(prob_4_csom)])
    
    %CSOM Softmax
    prob_1_earwax = (sum(ACTUAL_label(class_1)==PREDICTED_normal(class_4))/length(ACTUAL_label(class_4))*100);
    prob_2_earwax = (sum(ACTUAL_label(class_2)==PREDICTED_normal(class_4))/length(ACTUAL_label(class_4))*100);
    prob_3_earwax = (sum(ACTUAL_label(class_3)==PREDICTED_normal(class_4))/length(ACTUAL_label(class_4))*100);
    prob_4_earwax = (sum(ACTUAL_label(class_4)==PREDICTED_normal(class_4))/length(ACTUAL_label(class_4))*100);
    
    disp('Earwax Softmax')
    disp(['normal=' num2str(prob_1_earwax)])
    disp(['aom=' num2str(prob_2_earwax)])
    disp(['csom=' num2str(prob_3_earwax)])
    disp(['earwax=' num2str(prob_4_earwax)])
    
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
    
        %PIE CHARTS for SOFTMAX
        
        normal_pie = [(prob_1_normal/100) (prob_2_normal/100) (prob_3_normal/100) (prob_4_normal/100)];
        aom_pie = [(prob_1_aom/100) (prob_2_aom/100) (prob_3_aom/100) (prob_4_aom/100)];
        csom_pie = [(prob_1_csom/100) (prob_2_csom/100) (prob_3_csom/100) (prob_4_csom/100)];
        earwax_pie = [(prob_1_earwax/100) (prob_2_earwax/100) (prob_3_earwax/100) (prob_4_earwax/100)];
    
        X = categorical({ 'Normal SM' , 'AOM SM' , 'CSOM SM' , 'Earwax SM' });
        all_pie = [normal_pie; aom_pie; csom_pie; earwax_pie;];
        figure;
        b = barh(X, all_pie, 'BarWidth' , 1)
        
        xtips1 = b(1).YEndPoints + 0.01;
        ytips1 = b(1).XEndPoints;
        labels1 = string(b(1).YData);
        text(xtips1,ytips1,labels1, 'VerticalAlignment' , 'middle' )
    
        xtips2 = b(2).YEndPoints + 0.01;
        ytips2 = b(2).XEndPoints;
        labels2 = string(b(2).YData);
        text(xtips2,ytips2,labels2, 'VerticalAlignment' , 'middle' )
    
        xtips3 = b(3).YEndPoints + 0.01;
        ytips3 = b(3).XEndPoints;
        labels3 = string(b(3).YData);
        text(xtips3,ytips3,labels3, 'VerticalAlignment' , 'middle' )
    
        xtips4 = b(4).YEndPoints + 0.01;
        ytips4 = b(4).XEndPoints;
        labels4 = string(b(4).YData);
        text(xtips4,ytips4,labels4, 'VerticalAlignment' , 'middle' )
    
        legend('Normal', 'AOM', 'CSOM', 'Earwax')
        clear title; %clears previous title of confusion matrix
        title('Softmax Probabilities for Each Classification')
        xlabel('Probability')
        ylabel('Softmax Classfications')
        
        %bar(all_pie)
        
    
    title_cat={'''accuracy','''AUC_OVERALL','''AUC_AOM','''AUC_CSOM','''AUC_EARWAX', ...
           '''AUC_NORMAL','''sensitivity','''specificity','''precision','''recall', '''f_measure','''gmean', ...
           '''Normal-Normal', '''Normal-AOM', '''Normal-CSOM', '''Normal-Earwax', ...
           '''AOM-Normal', '''AOM-AOM', '''AOM-CSOM', '''AOM-Earwax', ...
           '''CSOM-Normal', '''CSOM-AOM', '''CSOM-CSOM', '''CSOM-Earwax', ...
           '''Earwax-Normal', '''Earwax-AOM', '''Earwax-CSOM', '''Earwax-Earwax'};
    VALUES=[accuracy, AUC_OVERALL,AUC,AUC2,AUC3,AUC4,sensitivity,specificity,precision,recall,f_measure,gmean, ...
            prob_1_normal, prob_2_normal,prob_3_normal, prob_4_normal, ...
            prob_1_aom, prob_2_aom,prob_3_aom, prob_4_aom, ...
            prob_1_csom, prob_2_csom,prob_3_csom, prob_4_csom, ...
            prob_1_earwax, prob_2_earwax,prob_3_earwax, prob_4_earwax];
    
    filename='performance.xlsx';
    xlswrite(filename,title_cat,'Sheet1','A117'); %Always change to empty cell
    xlswrite(filename,VALUES,'Sheet1', 'A118'); %Always change to empty cell
    %winopen(filename);  
    %end
    %winopen(filename);
toc
timeElapsed = toc

 