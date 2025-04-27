%% Experiment 2: CNN (Whole Image) - Using imagePretrainedNetwork (ResNet-18)
%% Clean environment

close all;
clear variables;
clc;
rng(42);
%% Set folder path

folder = "C:\Users\u3253992\workspace\computer-vision\assignment_1\u3253992_assignment_1\data\CUB_200_2011_Subset20classes\";
%% Load dataset

[trainingImageNames, validationImageNames, testImageNames, classNames, imageClassLabels] = loadDataset(folder);
numClasses = height(classNames);
%% Create datastores

disp('Training set class distribution:');
trainingImageDS = createImageDatastore(trainingImageNames, folder);

disp('Validation set class distribution:');
validationImageDS = createImageDatastore(validationImageNames, folder);

disp('Test set class distribution:');
testImageDS = createImageDatastore(testImageNames, folder);
%% Resize image datastores and combine them with the labels

targetSize = [224, 224];   % ResNet18 input size

trainCDS = resizeCombineDatastore(trainingImageDS, targetSize);
valCDS = resizeCombineDatastore(validationImageDS, targetSize);
testCDS = resizeCombineDatastore(testImageDS, targetSize);
%% Show example images

showImage(trainCDS)
%% Enables GPU

[device, useGPU] = enableGPU();
%% Load pretrained network using imagePretrainedNetwork

net = imagePretrainedNetwork("resnet18", Weights="none", NumClasses=numClasses);

net.Layers
%% Training options

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...
    'MiniBatchSize', 48, ...
    'MaxEpochs', 20, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', valCDS, ...
    'ValidationFrequency', 5, ...
    'Verbose', true, ...
    'Metrics',"accuracy", ...
    'Plots', 'training-progress');
%% Train the network

trainedNet = trainnet(trainCDS, net, "crossentropy", options);
%% Evaluate the model

scoresTest = minibatchpredict(trainedNet, testCDS)
pred = scores2label(scoresTest,classNames.className);
[~, yPred] = max(scoresTest, [], 2);
% disp("Predictions");
% disp(yPred);
yTest = str2double(string(testImageDS.Labels))
accuracy = sum(yPred == yTest) / numel(yTest);
disp(accuracy)
%% Plot confusion matrix

plotConfusionMatrix(yTest, yPred, "Exp.2 ResNet-18 Overall Accuracy: " + string(round(accuracy*100, 1)) + "%");