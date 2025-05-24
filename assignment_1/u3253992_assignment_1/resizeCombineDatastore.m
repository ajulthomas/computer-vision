function cds = resizeCombineDatastore(imageDS, targetSize)
    if nargin < 2
        targetSize = [128, 128];
        % targetSize = [224, 224];
        % targetSize = [256, 256];
        % targetSize = [224, 299];
        % targetSize = [384, 384];
    end
    % resized datastore
    rds = transform(imageDS, @(x) imresize(x, targetSize));
    labels = arrayDatastore(imageDS.Labels);
    cds = combine(rds, labels);
end