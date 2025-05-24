% normalizeFeatures.m
function [normFeatures, mu, sigma] = normalizeFeatures(features, mu, sigma)
    if nargin == 1
        mu = mean(features);
        sigma = std(features);
    end
    normFeatures = (features - mu) ./ sigma;
end