% showImage.m
function showImageTile(imageDS)
    reset(imageDS);
    figure(1)
    img = imageDS.read{1};
    subplot(1, 2, 1);
    imshow(img);
    title('Sample Image Resized');
end
