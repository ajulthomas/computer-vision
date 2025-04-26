function [trainingImageNames, validationImageNames, testImageNames, classNames, imageClassLabels, boundingBoxes] = loadDataset(folder, bb)
% loadDataset loads dataset file tables
% 
% If bb == true, also loads bounding box information.
%
% USAGE:
%   [trainNames, valNames, testNames, classNames, labels] = loadDataset(folder);
%   [trainNames, valNames, testNames, classNames, labels, boundingBoxes] = loadDataset(folder, true);

    if nargin < 2
        bb = false;  % Default: if bb not given, set it to false
    end

    % Load basic data
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

    % Load bounding boxes if requested
    if bb
        boundingBoxes = readtable(fullfile(folder, "bounding_boxes.txt"), 'ReadVariableNames', false);
        boundingBoxes.Properties.VariableNames = {'index', 'x', 'y', 'w', 'h'};
    else
        boundingBoxes = [];  % Return empty if not requested
    end
end
