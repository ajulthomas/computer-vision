% createImageDatastore.m
function imageDS = createImageDatastore(imageNames, folder, bb, boundingBoxes)
    
    % if bounding box, create image-box map
    
    if nargin < 3
        bb = false;  % Default: if bb not given, set it to false
        boundingBoxes = [];
    end

    filePaths = strings(height(imageNames), 1);
    for i = 1:height(imageNames)
        filePaths(i) = fullfile(folder, "images", string(cell2mat(imageNames.imageName(i))));
    end


    imageDS = imageDatastore(filePaths, 'labelSource', 'foldernames', 'FileExtensions', {'.jpg'});

    if bb
        image_box_map = returnMapping(imageNames, boundingBoxes);
        imageDS.ReadFcn = @(filename) readImagesIntoDatastoreBB_Fast(filename, image_box_map);
    else
        imageDS.ReadFcn = @readImagesIntoDatastore;
    end
    disp(countEachLabel(imageDS));
end

%% Helper function mapping image names to bounding boxes and vice versa

function image_box_map = returnMapping(ImageNames, boundingBoxes)
    image_box_map = containers.Map;
    for i = 1:size(ImageNames, 1)
        fn = ImageNames{i,2}{1};
        fn = split(fn, "\");
        fn = split(fn, "/");
        image_box_map(fn{end}) = [boundingBoxes{ImageNames{i,1}, 2:5}];
    end
end