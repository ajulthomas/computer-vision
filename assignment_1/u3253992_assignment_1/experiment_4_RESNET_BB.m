%% Experiment 4: Transfer Learning with ResNet-50
%% Clean environment
close all; clear variables; clc; rng(42);

%% Set folder path
folder = "C:\Users\u3253992\workspace\computer-vision\assignment_1\u3253992_assignment_1\data\CUB_200_2011_Subset20classes\";

%% Load dataset
[trainingImageNames, validationImageNames, testImageNames, classNames, imageClassLabels, boundingBoxes] = ...
    loadDataset(folder, true);                                                    % :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
numClasses = height(classNames);

%% Create datastores
trainingImageDS   = createImageDatastore(trainingImageNames,   folder, true, boundingBoxes);
validationImageDS = createImageDatastore(validationImageNames, folder, true, boundingBoxes);
testImageDS       = createImageDatastore(testImageNames,       folder, true, boundingBoxes);

%% Load pretrained ResNet-50
net       = resnet50;                                                            % requires Deep Learning Toolbox Model for ResNet-50
inputSize = net.Layers(1).InputSize;                                             % [224 224 3]

%% Resize & combine
targetSize = inputSize(1:2);
trainCDS = resizeCombineDatastore(trainingImageDS,   targetSize);
valCDS   = resizeCombineDatastore(validationImageDS, targetSize);
testCDS  = resizeCombineDatastore(testImageDS,       targetSize);

%% Optional: preview
showImage(trainCDS)

%% Enable GPU
[device, useGPU] = enableGPU();

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

%% Training options
options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',valCDS, ...
    'ValidationFrequency',floor(numel(trainingImageDS.Files)/32), ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','gpu');

%% Train
resnet20 = trainNetwork(trainCDS, lgraph, options);

%% Evaluate
[trainAcc, ~]   = evaluateClassifier(resnet20, trainCDS,   trainingImageDS.Labels,   true);
[valAcc,   ~]   = evaluateClassifier(resnet20, valCDS,     validationImageDS.Labels, true);
[testAcc, preds] = evaluateClassifier(resnet20, testCDS,   testImageDS.Labels,       true);

fprintf("Train Accuracy = %.1f%%\n", trainAcc*100);
fprintf("Val   Accuracy = %.1f%%\n", valAcc*100);
fprintf("Test  Accuracy = %.1f%%\n", testAcc*100);

%% Confusion matrix
plotConfusionMatrix(testImageDS.Labels, preds, ...
    sprintf("ResNet-50 Transfer Learning: %.1f%%", testAcc*100));






































% %% Experiment 4: Transfer Learning with ResNet-50
% %% Clean environment
% 
% close all;
% clear variables;
% clc;
% rng(42);
% %% Set folder path
% 
% folder = "C:\Users\u3253992\workspace\computer-vision\assignment_1\u3253992_assignment_1\data\CUB_200_2011_Subset20classes\";
% %% Load dataset
% 
% [trainingImageNames, validationImageNames, testImageNames, classNames, imageClassLabels, boundingBoxes] = loadDataset(folder, true);
% numClasses = height(classNames);
% %% Create datastores
% 
% disp('Training set class distribution:');
% trainingImageDS = createImageDatastore(trainingImageNames, folder, true, boundingBoxes);
% 
% disp('Validation set class distribution:');
% validationImageDS = createImageDatastore(validationImageNames, folder, true, boundingBoxes);
% 
% disp('Test set class distribution:');
% testImageDS = createImageDatastore(testImageNames, folder, true, boundingBoxes);
% %% Load pretrained ResNet-50
% 
% net = resnet50;                             % requires Deep Learning Toolbox Model for ResNet-50 Network
% inputSize = net.Layers(1).InputSize;        % typically [224 224 3]
% %% Resize image datastores and combine them with the labels
% 
% targetSize = inputSize(1:2);
% trainCDS = resizeCombineDatastore(trainingImageDS, targetSize);
% valCDS = resizeCombineDatastore(validationImageDS, targetSize);
% testCDS = resizeCombineDatastore(testImageDS, targetSize);
% %% Show images
% 
% showImage(trainCDS)
% %% Enables GPU
% 
% [device, useGPU] = enableGPU();
% %% CNN - Convolutional Neural Network
% %% Replace final layers to match numClasses
% 
% lgraph = layerGraph(net);
% %%
% lgraph.Layers
% %%
% %% Find the layers to replace
% [learnableLayer, classLayer] = findLayersToReplace(lgraph);
% 
% %% Create new layers
% newLearnableLayer = fullyConnectedLayer(numClasses, ...
%     'Name','new_fc', ...
%     'WeightLearnRateFactor',10, ...
%     'BiasLearnRateFactor',10);
% 
% newClassLayer = classificationLayer('Name','new_classoutput');
% 
% %% Replace the layers
% lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);
% lgraph = replaceLayer(lgraph, classLayer.Name,     newClassLayer);
% %%
% 
% % % 1) Replace fully connected layer
% % newFCLayer = fullyConnectedLayer(numClasses, ...
% %     'Name','fc20', ...
% %     'WeightLearnRateFactor',10, ...
% %     'BiasLearnRateFactor',10);
% % lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);
% % 
% % % 2) Replace softmax
% % newSoftmax = softmaxLayer('Name','softmax20');
% % lgraph = replaceLayer(lgraph, 'fc1000_softmax', newSoftmax);
% % 
% % % 3) Replace classification output layer
% % newClassLayer = classificationLayer('Name','classoutput20');
% % lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);
% %% Training options
% 
% options = trainingOptions('adam', ...
%     'MiniBatchSize',32, ...
%     'MaxEpochs',6, ...
%     'InitialLearnRate',1e-4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',valCDS, ...
%     'ValidationFrequency',floor(numel(trainingImageDS.Files)/32), ...
%     'Verbose',true, ...
%     'Plots','training-progress', ...
%     'ExecutionEnvironment','gpu');          % uses GPU if available
% %% Train the network
% 
% resnet20 = trainNetwork(trainCDS, lgraph, options);
% %% Evaluate the model
% 
% [trainAcc, ~] = evaluateClassifier(resnet20, trainCDS,   trainingImageDS.Labels,   true);
% [valAcc,   ~] = evaluateClassifier(resnet20, valCDS,     validationImageDS.Labels, true);
% [testAcc,  preds] = evaluateClassifier(resnet20, testCDS, testImageDS.Labels,       true);
% 
% 
% fprintf("Train Accuracy = %.1f%%\n", trainAcc*100);
% fprintf("Val   Accuracy = %.1f%%\n", valAcc*100);
% fprintf("Test  Accuracy = %.1f%%\n", testAcc*100);
% %% Confusion matrix
% 
% plotConfusionMatrix(testImageDS.Labels, preds, ...
%     sprintf("ResNet-50 Transfer Learning: %.1f%%", testAcc*100));
% %% 
% %