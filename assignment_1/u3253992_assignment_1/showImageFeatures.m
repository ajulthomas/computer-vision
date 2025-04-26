% showImageFeatures.m
function showImageFeatures(imageDS)
    figure(1)
    img = imageDS.read{1};
    SIFTpoints = detectSIFTFeatures(rgb2gray(img));
    subplot(1, 2, 1);
    imshow(img);
    title('Sample Image Resized');
    subplot(1, 2, 2);
    imshow(rgb2gray(img)); hold on;
    title('SIFT Feature Points');
    plot(SIFTpoints.selectStrongest(50));
    hold off;
end
