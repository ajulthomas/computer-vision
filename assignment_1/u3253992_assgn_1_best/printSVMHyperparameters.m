% printSVMHyperparameters.m
function printSVMHyperparameters(Classifier)
% printSVMHyperparameters - Display hyperparameters of SVM ECOC Classifier
%
% INPUT:
%   Classifier - Trained fitcecoc model containing SVM binary learners
%
% USAGE:
%   printSVMHyperparameters(Classifier)

    if ~isa(Classifier, 'ClassificationECOC')
        error('Input must be a ClassificationECOC model.');
    end

    disp("SVM Classifier Hyperparameters:");

    numLearners = numel(Classifier.BinaryLearners);

    for i = 1:numLearners
        learner = Classifier.BinaryLearners{i};  % Extract each binary SVM model
        
        fprintf('--- Binary Learner #%d ---\n', i);

        if isprop(learner, 'KernelFunction')
            disp(['Kernel Function: ' learner.KernelFunction]);
        end

        if isprop(learner, 'BoxConstraints')
            disp(['Box Constraint (C): ' num2str(learner.BoxConstraints(1))]);
        end

        if isprop(learner, 'KernelParameters') && isfield(learner.KernelParameters, 'Scale')
            disp(['Kernel Scale (if RBF): ' num2str(learner.KernelParameters.Scale)]);
        end

        if isprop(learner, 'Solver')
            disp(['Solver: ' learner.Solver]);
        end
        
        disp('-----------------------------');
    end
end