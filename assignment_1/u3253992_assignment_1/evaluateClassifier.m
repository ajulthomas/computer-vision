% evaluateClassifier.m
function [accuracy, predictions] = evaluateClassifier(classifier, testFeatures, testLabels, nn)
    if nargin < 4
        nn=false;
    end
    if nn
        predictions = classify(classifier, testFeatures);
    else
        predictions = predict(classifier, testFeatures);
    end
    accuracy = sum(predictions == testLabels) / numel(testLabels);
end