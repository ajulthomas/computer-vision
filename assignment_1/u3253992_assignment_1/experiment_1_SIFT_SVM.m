% Main script to run Experiment 1: SIFT + SVM (Whole Image)
%% Clean environment

close all;
clear variables;
clc;
%% Set folder path

folder = "C:\Users\u3253992\workspace\computer-vision\assignment_1\u3253992_assignment_1\data\CUB_200_2011_Subset20classes\";
%% Load dataset

[trainingImageNames, validationImageNames, testImageNames, classNames, imageClassLabels] = loadDataset(folder);
%% Create datastores

disp('Training set class distribution:');
trainingImageDS = createImageDatastore(trainingImageNames, folder);

disp('Validation set class distribution:');
validationImageDS = createImageDatastore(validationImageNames, folder);

disp('Test set class distribution:');
testImageDS = createImageDatastore(testImageNames, folder);

%% Resize image datastores and combine them with the labels

trainCDS = resizeCombineDatastore(trainingImageDS);
valCDS = resizeCombineDatastore(validationImageDS);
testCDS = resizeCombineDatastore(testImageDS);
%% Show images

showImageFeatures(trainCDS)
%% Extract SIFT features

numFeatures = 50;
maxFeatures = 100;

[trainingFeatures, trainingLabels] = extractSIFTFeaturesFromDatastore(trainingImageDS, numFeatures, maxFeatures);
[testFeatures, testLabels] = extractSIFTFeaturesFromDatastore(testImageDS, numFeatures, maxFeatures);
%% Enables GPU

[device, useGPU] = enableGPU();
%% Train classifier

SVMClassifier = trainSVMClassifier(trainingFeatures, trainingLabels);
%% Evaluate

[accuracy, predictions] = evaluateClassifier(SVMClassifier, testFeatures, testLabels);
%% Plot confusion matrix

plotConfusionMatrix(testLabels, predictions, "Overall Accuracy (SIFT): " + string(round(accuracy*100, 1)) + "%");
%% 
%