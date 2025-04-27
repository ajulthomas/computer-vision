function SVMClassifier = trainSVMClassifier(trainingFeatures, trainingLabels)
    t = templateSVM('KernelFunction', 'rbf', 'Standardize', true, 'Type','classification', 'Solver','SMO', 'KernelScale','auto');
    options = struct('Optimizer', 'bayesopt', 'ShowPlots', true, 'UseParallel', true, 'CVPartition', cvpartition(trainingLabels, 'KFold', 5));
    SVMClassifier = fitcecoc(trainingFeatures, trainingLabels, 'Learners', t, ...
        'OptimizeHyperparameters', {'BoxConstraint'}, ...
        'HyperparameterOptimizationOptions', options);
end
