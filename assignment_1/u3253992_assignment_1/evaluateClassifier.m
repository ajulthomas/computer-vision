% evaluateClassifier.m
function [accuracy, predictions] = evaluateClassifier(classifier, testFeatures, testLabels)
    predictions = predict(classifier, testFeatures);
    accuracy = sum(predictions == testLabels) / numel(testLabels);
end