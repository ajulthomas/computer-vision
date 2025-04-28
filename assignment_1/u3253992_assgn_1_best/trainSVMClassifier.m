function SVMClassifier = trainSVMClassifier(trainingFeatures, trainingLabels)
    t = templateSVM('KernelFunction', 'gaussian', 'Standardize', true, 'Type','classification', 'Solver','SMO', 'KernelScale','auto'); 
    SVMClassifier = fitcecoc(trainingFeatures, trainingLabels, 'Learners', t,'Coding','onevsall');
end

 % options = struct('Optimizer', 'bayesopt', 'ShowPlots', true, 'UseParallel', true);
% 'OptimizeHyperparameters', {'BoxConstraint'}, ...
% 'HyperparameterOptimizationOptions', options
% 'BoxConstraint',0.93985