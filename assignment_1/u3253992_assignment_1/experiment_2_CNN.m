%% Experiment 2: CNN (Whole Image)
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

% targetSize = [128, 128];
targetSize = [224, 224];
% targetSize = [256, 256];
% targetSize = [229, 299];
% targetSize = [384, 384];

trainCDS = resizeCombineDatastore(trainingImageDS, targetSize);
valCDS = resizeCombineDatastore(validationImageDS, targetSize);
testCDS = resizeCombineDatastore(testImageDS, targetSize);
%% Show images

showImageFeatures(trainCDS)
%% Enables GPU

[device, useGPU] = enableGPU();
%% CNN - Convolutional Neural Network

% defines CNN layers
layers = [
    % imageInputLayer([128 128 3])    % This needs to match the image size
    imageInputLayer([224 224 3])
    % imageInputLayer([256 256 3])
    % imageInputLayer([229 229 3])
    % imageInputLayer([384 384 3])

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(20)
    softmaxLayer
    classificationLayer];

% sets training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MiniBatchSize', 50, ...
    'MaxEpochs', 10, ...
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 1, ...
    'ValidationData', valCDS, ...
    'Plots','training-progress');

% model training
simpleCNN = trainNetwork(trainCDS, layers, options);

%% Evaluate the model

[trainAccuracy, trainPredictions] = evaluateClassifier(simpleCNN, trainCDS, trainingImageDS.Labels, true);
[valAccuracy, valPredictions] = evaluateClassifier(simpleCNN, valCDS, validationImageDS.Labels, true);
[accuracy, predictions] = evaluateClassifier(simpleCNN, testCDS, testImageDS.Labels, true);


disp("Train Accuracy = "+string(round(trainAccuracy*100, 1)))
disp("Validation Accuracy = "+string(round(valAccuracy*100, 1)))
disp("Test Accuracy = "+string(round(accuracy*100, 1)))
%% Plot confusion matrix

plotConfusionMatrix(testImageDS.Labels, predictions, "Exp.2 Overall Accuracy (CNN): " + string(round(accuracy*100, 1)) + "%");
%% 
%