% trainNaiveBayesClassifier.m
function NaiveBayesModel = trainNaiveBayesClassifier(trainFeatures, trainLabels, kfold)
% trainNaiveBayesClassifier - Train a Naive Bayes classifier with hyperparameter tuning and k-fold cross-validation
%
% INPUTS:
%   trainFeatures - Matrix of extracted features (numSamples x numFeatures)
%   trainLabels   - Corresponding labels (categorical, numeric, or string array)
%   kfold         - Number of folds for cross-validation (e.g., 5, 10)
%
% OUTPUT:
%   NaiveBayesModel - Trained Naive Bayes model (ClassificationNaiveBayes)

    if nargin < 3
        kfold = 5;  % Default to 5-fold CV if not provided
    end

    % Set up cross-validation partition
    cvPartition = cvpartition(trainLabels, 'KFold', kfold);

    % Define hyperparameter tuning options
    optimizationOptions = struct(...
        'AcquisitionFunctionName', 'expected-improvement-plus', ...
        'ShowPlots', true, ...
        'UseParallel', true, ...   % Use true if you have parallel computing toolbox
        'CVPartition', cvPartition, ...  % <<< Use k-fold CV during hyperparameter search!
        'MaxObjectiveEvaluations', 30);  % Number of parameter settings to try

    % Train Naive Bayes classifier with optimization
    NaiveBayesModel = fitcnb(trainFeatures, trainLabels, ...
        'OptimizeHyperparameters', {'DistributionNames', 'Width'}, ...
        'HyperparameterOptimizationOptions', optimizationOptions);
end