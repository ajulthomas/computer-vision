% showImageFeatures.m
function hogFeatureSize = showImageFeatures(imageDS, hog, cellSize)
    if nargin < 2
        hog = false;
        cellSize = [];
    end
    reset(imageDS);
    hogFeatureSize = 0;
    
    figure(1)
    img = imageDS.read{1};
    subplot(1, 2, 1);
    imshow(img);
    title('Sample Image Resized');

    if hog
        [hog, vis] = extractHOGFeatures(img,'CellSize', cellSize);
        hogFeatureSize = length(hog);
        subplot(1, 2, 2);
        plot(vis);
        titleStr = "HOG CellSize =" + mat2str(cellSize);
        title({titleStr; ['Length = ' num2str(length(hog))]});
    else
        SIFTpoints = detectSIFTFeatures(rgb2gray(img));
        subplot(1, 2, 2);
        imshow(rgb2gray(img)); hold on;
        title('SIFT Feature Points');
        plot(SIFTpoints.selectStrongest(50));
    end

    hold off;
end
