% plotConfusionMatrix.m
function cm = plotConfusionMatrix(trueLabels, predLabels, titleText)
    figure;
    [m, order] = confusionmat(trueLabels, predLabels);
    cm = confusionchart(m, order, 'ColumnSummary', 'column-normalized', 'RowSummary', 'row-normalized');
    title(titleText);
    classwisePosRecog = zeros(height(order), 1);
    samplesPerRow = sum(m, 2);
    for i = 1:height(order)
        classwisePosRecog(i) = round(100 * m(i, i) / samplesPerRow(i), 1);
    end
    disp('Classwise Recognition Rates:');
    disp(classwisePosRecog)
end