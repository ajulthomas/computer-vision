function cds = resizeCombineDatastore(imageDS)
    targetSize = [224, 224];
    % resized datastore
    rds = transform(imageDS, @(x) imresize(x, targetSize));
    labels = arrayDatastore(imageDS.Labels);
    cds = combine(rds, labels);
end