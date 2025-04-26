% plotConfusionMatrix.m
function plotConfusionMatrix(trueLabels, predLabels, titleText)
    figure;
    [m, order] = confusionmat(trueLabels, predLabels);
    cm = confusionchart(m, order, 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
    title(titleText);
end