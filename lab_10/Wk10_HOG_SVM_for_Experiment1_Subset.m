% Week 10 example code for training a simple SVM classifier on the
% CUB_200_2011_Subset20classes dataset using HOG features.
%
% Author: Roland Goecke
% Date created: 26/04/2022
% Modified by Ibrahim Radwan 
% Date last updated: 6/04/25


close all;
clear variables;
clc;

%% Read the training, validation and test partitions from the relevant
%  text files. 
%  *** Adjust the file path as required. ***
folder = "C:\Dropbox\UC\Teaching\8890_CVIA_PG\2022\Data\CUB_200_2011_Subset20classes\";
%folder = "P:\CUB_200_2011_Subset20classes\";
%folder = "/Users/s423738/Dropbox/UC/Teaching/8890_CVIA_PG/2022/Data/CUB_200_2011_Subset20classes/";
trainingImageNames = readtable(fullfile(folder, "train.txt"), ... 
    'ReadVariableNames', false);
trainingImageNames.Properties.VariableNames = {'index', 'imageName'};

validationImageNames = readtable(folder + "validate.txt", ... 
    'ReadVariableNames', false);
validationImageNames.Properties.VariableNames = {'index', 'imageName'};

testImageNames = readtable(folder + "test.txt", ... 
    'ReadVariableNames', false);
testImageNames.Properties.VariableNames = {'index', 'imageName'};

%% Read class info from the relevant text files
classNames = readtable(folder + "classes.txt", ...
    'ReadVariableNames', false);
classNames.Properties.VariableNames = {'index', 'className'};

imageClassLabels = readtable(folder + "image_class_labels.txt", ...
    'ReadVariableNames', false);
imageClassLabels.Properties.VariableNames = {'index', 'classLabel'};

%% Create lists of image names for training, validation and test subsets.
%  To be precise, we create an array of strings containing the full file
%  path and file names for each data partition.
trainingImageList = strings(height(trainingImageNames), 1);
for iI = 1:height(trainingImageNames)
    trainingImageList(iI) = string(fullfile(folder, "images/", ...
        string(cell2mat(trainingImageNames.imageName(iI)))));
end

validationImageList = strings(height(validationImageNames), 1);
for iI = 1:height(validationImageNames)
    validationImageList(iI) = string(folder + "images/" + ...
        string(cell2mat(validationImageNames.imageName(iI))));
end

testImageList = strings(height(testImageNames), 1);
for iI = 1:height(testImageNames)
    testImageList(iI) = string(folder + "images/" + ...
        string(cell2mat(testImageNames.imageName(iI))));
end

%% Create image datastores for training, validation and test subsets
trainingImageDS = imageDatastore(trainingImageList, 'labelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});
trainingImageDS.ReadFcn = @readImagesIntoDatastore;
disp('Training set class distribution:');
countEachLabel(trainingImageDS)

validationImageDS = imageDatastore(validationImageList, 'labelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});
validationImageDS.ReadFcn = @readImagesIntoDatastore;
disp('Validation set class distribution:');
countEachLabel(validationImageDS)

testImageDS = imageDatastore(testImageList, 'labelSource', 'foldernames', ...
    'FileExtensions', {'.jpg'});
testImageDS.ReadFcn = @readImagesIntoDatastore;
disp('Test set class distribution:');
countEachLabel(testImageDS)

%% The images all have different spatial resolutions (width x height), so 
%  need to resize them to the same size. (Experiment with different sizes!)
targetSize = [100, 100];
trainingImageDS_Resized = transform(trainingImageDS, @(x) imresize(x,targetSize));
validationImageDS_Resized = transform(validationImageDS, @(x) imresize(x,targetSize));
testImageDS_Resized = transform(testImageDS, @(x) imresize(x,targetSize));

% Combine transformed datastores and labels
labelsTraining = arrayDatastore(trainingImageDS.Labels);
cdsTraining = combine(trainingImageDS_Resized, labelsTraining);
labelsValidation = arrayDatastore(validationImageDS.Labels);
cdsValidation = combine(validationImageDS_Resized, labelsValidation);
labelsTest = arrayDatastore(testImageDS.Labels);
cdsTest = combine(testImageDS_Resized, labelsTest);

%% Display a sample image from the datastore and HOG features
figure(1);
img = cdsTraining.read{1};
[hog_16x16, vis16x16] = extractHOGFeatures(img,'CellSize',[16 16]);
%[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
subplot(1, 2, 1); 
imshow(img);
title('Sample Image Resized');
subplot(1, 2, 2);
plot(vis16x16); 
%plot(vis8x8); 
title({'HOG CellSize = [16 16]'; ['Length = ' num2str(length(hog_16x16))]});
%title({'HOG CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

%% Extract HOG features
cellSize = [16 16];     % Set cell size to 16x16
hogFeatureSize = length(hog_16x16);
%cellSize = [8 8];     % Set cell size to 8x8
%hogFeatureSize = length(hog_8x8);

% Start by extracting HOG features from the training set. These features 
% will be used to train the classifier.
% Loop over the trainingSet and extract HOG features from each image.
numImagesTrain = numel(trainingImageDS_Resized.UnderlyingDatastores{1, 1}.Files);
trainingFeatures = zeros(numImagesTrain, hogFeatureSize, 'single');
reset(cdsTraining);     % Make sure we start with the first image
for i = 1:numImagesTrain
    imgFromDS = read(cdsTraining);    % Get item from datastore. Note, this returns a cell array.    
    imgGray = im2gray(imgFromDS{1});      % Ensure images are grayscale
    
    trainingFeatures(i, :) = extractHOGFeatures(imgGray, 'CellSize', cellSize);  
end

% Now extract HOG features from the test set as well.
numImagesTest = numel(testImageDS_Resized.UnderlyingDatastores{1, 1}.Files);
testFeatures = zeros(numImagesTest, hogFeatureSize, 'single');
reset(cdsTest);     % Make sure we start with the first image
for i = 1:numImagesTest
    imgFromDS = read(cdsTest);    % Get item from datastore
    imgGray = im2gray(imgFromDS{1});  % Ensure images are grayscale
    
    testFeatures(i, :) = extractHOGFeatures(imgGray, 'CellSize', cellSize);  
end


%% Check if we have a GPU available and clear any old data from it
if (gpuDeviceCount() > 0)
    disp('Found GPU:');
    disp(gpuDeviceTable);
    device = gpuDevice(1);
    reset(device);  % Clear previous values that might still be on the GPU
end

%% Train a multi-class SVM
t = templateLinear('Solver', 'dual');
%t = templateSVM('KernelFunction','linear');
options = struct('UseParallel', true);

% fitcecoc uses multiple SVM learners and a 'One-vs-One' encoding scheme.
Classifier = fitcecoc(trainingFeatures, trainingImageDS.Labels, 'Coding', 'onevsall', 'Learners', t, ...
    'OptimizeHyperparameters', {'Lambda'}, ...
    'HyperparameterOptimizationOptions', options);
% Classifier = fitcecoc(trainingFeatures, trainingImageDS.Labels, 'Learners', t, ...
%     'FitPosterior',true,'Verbose',2);

%% Test the accuracy on the test partition
YPred = predict(Classifier, testFeatures);
YTest = testImageDS.Labels;

% Calculate overall accuracy
accuracy = sum(YPred == YTest)/numel(YTest) % Output on command line

% Show confusion matrix in figure
[m, order] = confusionmat(YTest, YPred);
figure(2);
cm = confusionchart(m, order, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');
title("Overall Accuracy (HOG): "+ string(round(accuracy*100, 1)) +"%");
