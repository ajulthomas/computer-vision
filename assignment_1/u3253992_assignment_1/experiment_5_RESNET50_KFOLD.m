%% Experiment 4: Transfer Learning with ResNet-50
%% Clear Environment

close all;
clear variables;
existing_GUIs = findall(0);
if length(existing_GUIs) > 1
    delete(existing_GUIs);
end
clc;
%% Set folder path

folder = "C:\Users\u3253992\workspace\computer-vision\assignment_1\u3253992_assignment_1\data\CUB_200_2011_Subset20classes\";
%% Load All Images

imgFolder = folder + "images/";
imgTxtFolder = folder + "images.txt";
imageDS = imageDatastore(imgFolder, "IncludeSubfolders",true, "LabelSource","foldernames");
[f1, f2, f3, f4, f5] = splitEachLabel(imageDS, 0.2, 0.2, 0.2, 0.2);

numFolds = 5;
numClasses = 20;
%% Load pretrained ResNet-50
% requires Deep Learning Toolbox Model for ResNet-50

net       = resnet50;                                
inputSize = net.Layers(1).InputSize;
targetSize = inputSize(1:2);
%% Build the layer graph and swap out the last 3 layers

lgraph = layerGraph(net);

% 1) New fully‐connected layer (20 classes)
newFc = fullyConnectedLayer(numClasses, ...
    'Name','fc20', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

% 2) New softmax
newSoftmax = softmaxLayer('Name','softmax20');

% 3) New classification output
newClassOutput = classificationLayer('Name','classoutput20');

% Replace them
lgraph = replaceLayer(lgraph, 'fc1000', newFc);
lgraph = replaceLayer(lgraph, 'fc1000_softmax', newSoftmax);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassOutput);
%% Enable GPU

[device, useGPU] = enableGPU();
%% Train the simple CNN model for each fold

accuracy_overall = 0.0;
for i = 1:numFolds
    [cdsTraining, cdsValidation, cdsTest, trainingImageDS, validationImageDS, testImageDS] = ...
        getFoldsFor5FoldCrossVal(i, f1, f2, f3, f4, f5, folder, imgTxtFolder, targetSize);

    % Set the training options
    options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',cdsValidation, ...
    'ValidationFrequency',floor(numel(trainingImageDS.Files)/32), ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','gpu');

    resnet20 = trainNetwork(cdsTraining, lgraph, options);

    YPred = classify(resnet20, cdsTest);
    YTest = testImageDS.Labels;
    
    accuracy = sum(YPred == YTest)/numel(YTest); % Output on command line
    disp("Accuracy for Run "+ string(i)+" is: " + accuracy);

    % Show confusion matrix in figure
    [m, order] = confusionmat(YTest, YPred);
    figure(i);
    cm = confusionchart(m, order, ...
        'ColumnSummary','column-normalized', ...
        'RowSummary','row-normalized');
    title("Overall Accuracy for Run "+ string(i)+" : "+ ...
        string(round(accuracy*100, 1)) +"%");

    accuracy_overall = accuracy_overall+accuracy;
end

disp("Average accuracy of five folds is "+ string(accuracy_overall/numFolds))