%extractHOGFeaturesFromDatastore.m
function features = extractHOGFeaturesFromDatastore(cds, hogFeatureSize, cellSize)
% extractHOGFeaturesFromDatastore extracts HOG features from an image datastore
%
% INPUTS:
%   cds             - combined datastore (cropped and resized images + labels)
%   hogFeatureSize  - size of HOG feature vector (length of a sample HOG feature)
%   cellSize        - HOG cell size [height width]
%
% OUTPUT:
%   features        - extracted HOG feature matrix (numImages x hogFeatureSize)

    numImages = numel(cds.UnderlyingDatastores{1, 1}.UnderlyingDatastores{1, 1}.Files);
    features = zeros(numImages, hogFeatureSize, 'single');

    reset(cds);  % Reset datastore
    for i = 1:numImages
        imgFromDS = read(cds);       % Read one item (cell array)
        imgGray = im2gray(imgFromDS{1});  % Convert to grayscale
        features(i, :) = extractHOGFeatures(imgGray, 'CellSize', cellSize);
    end
end
