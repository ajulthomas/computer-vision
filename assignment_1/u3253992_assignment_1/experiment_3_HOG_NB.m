% Main script to run Experiment 3: HOG + SVM (Whole Image)
%% Clean environment

close all;
clear variables;
clc;
%% Set folder path

folder = "C:\Users\u3253992\workspace\computer-vision\assignment_1\u3253992_assignment_1\data\CUB_200_2011_Subset20classes\";
%% Load dataset

[trainingImageNames, validationImageNames, testImageNames, classNames, imageClassLabels, boundingBoxes] = loadDataset(folder, true);
%% Create datastores

disp('Training set class distribution:');
trainingImageDS = createImageDatastore(trainingImageNames, folder, true, boundingBoxes);

disp('Validation set class distribution:');
validationImageDS = createImageDatastore(validationImageNames, folder, true, boundingBoxes);

disp('Test set class distribution:');
testImageDS = createImageDatastore(testImageNames, folder, true, boundingBoxes);

%% Resize image datastores and combine them with the labels

trainCDS = resizeCombineDatastore(trainingImageDS);
valCDS = resizeCombineDatastore(validationImageDS);
testCDS = resizeCombineDatastore(testImageDS);
%% Show images

cellSize = [16 16]
hogFeatureSize = showImageFeatures(trainCDS, true, cellSize)
disp("HOG feature size is " + hogFeatureSize)
%% Extract HOG Features

trainFeatures = extractHOGFeaturesFromDatastore(trainCDS, hogFeatureSize, cellSize)
testFeatures = extractHOGFeaturesFromDatastore(testCDS, hogFeatureSize, cellSize)
%% Enables GPU

[device, useGPU] = enableGPU();
%% Train classifier

NBClassifier = trainNBClassifier(trainFeatures, trainingImageDS.Labels);
%% Evaluate

[accuracy, predictions] = evaluateClassifier(NBClassifier, testFeatures, testImageDS.Labels);
%%
[trainAccuracy, trainPredictions] = evaluateClassifier(NBClassifier, trainFeatures, trainingImageDS.Labels);

disp("Train Accuracy = "+string(round(trainAccuracy*100, 1)))
disp("Test Accuracy = "+string(round(accuracy*100, 1)))
%% Plot confusion matrix

plotConfusionMatrix(testImageDS.Labels, predictions, "Overall Accuracy (Experiment_2 HOG - NB): " + string(round(accuracy*100, 1)) + "%");
%% 
%