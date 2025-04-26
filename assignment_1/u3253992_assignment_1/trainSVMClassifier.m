function SVMClassifier = trainSVMClassifier(trainingFeatures, trainingLabels)
    t = templateSVM('KernelFunction', 'rbf', 'Standardize', false);
    options = struct('Optimizer', 'bayesopt', 'ShowPlots', true, 'UseParallel', true);
    SVMClassifier = fitcecoc(trainingFeatures, trainingLabels, 'Learners', t, ...
        'OptimizeHyperparameters', {'BoxConstraint', 'KernelScale'}, ...
        'HyperparameterOptimizationOptions', options);
end