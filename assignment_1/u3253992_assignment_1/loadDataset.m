% loadDataset.m
function [trainingImageNames, validationImageNames, testImageNames, classNames, imageClassLabels] = loadDataset(folder)
    trainingImageNames = readtable(fullfile(folder, "train.txt"), 'ReadVariableNames', false);
    trainingImageNames.Properties.VariableNames = {'index', 'imageName'};

    validationImageNames = readtable(fullfile(folder, "validate.txt"), 'ReadVariableNames', false);
    validationImageNames.Properties.VariableNames = {'index', 'imageName'};

    testImageNames = readtable(fullfile(folder, "test.txt"), 'ReadVariableNames', false);
    testImageNames.Properties.VariableNames = {'index', 'imageName'};

    classNames = readtable(fullfile(folder, "classes.txt"), 'ReadVariableNames', false);
    classNames.Properties.VariableNames = {'index', 'className'};

    imageClassLabels = readtable(fullfile(folder, "image_class_labels.txt"), 'ReadVariableNames', false);
    imageClassLabels.Properties.VariableNames = {'index', 'classLabel'};
end