% createImageDatastore.m
function imageDS = createImageDatastore(imageNames, folder)
    filePaths = strings(height(imageNames), 1);
    for i = 1:height(imageNames)
        filePaths(i) = fullfile(folder, "images", string(cell2mat(imageNames.imageName(i))));
    end
    imageDS = imageDatastore(filePaths, 'labelSource', 'foldernames', 'FileExtensions', {'.jpg'});
    imageDS.ReadFcn = @readImagesIntoDatastore;
    countEachLabel(imageDS);
end