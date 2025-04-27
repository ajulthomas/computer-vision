% trainNBClassifier.m
function NaiveBayesModel = trainNBClassifier(trainFeatures, trainLabels)
% trainNaiveBayesClassifier - Train a Naive Bayes classifier with hyperparameter tuning and k-fold cross-validation
%
% INPUTS:
%   trainFeatures - Matrix of extracted features (numSamples x numFeatures)
%   trainLabels   - Corresponding labels (categorical, numeric, or string array)

    % Train Naive Bayes classifier with optimization
    NaiveBayesModel = fitcnb(trainFeatures, trainLabels, "DistributionNames","normal", "Standardize", true);
end