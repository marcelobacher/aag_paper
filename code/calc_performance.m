function [f1, auc, w, tpr, fpr] = calc_performance(yhat, ...
  y_weights, labels, nVal, nTest, nAnomaly)

try
  y = bsxfun(@eq, yhat, labels);
catch e
  disp(e);
end
% use only validation weights of subspaces
% % y_weights = sum(y(1:nVal,:),1)/nVal;
y_weighted = sum(bsxfun(@times, y, y_weights),2);
y_weighted_sorted = sort(y_weighted(1:nVal),'ascend');
cut = mean(y_weighted_sorted)-[3 2 1]*std(y_weighted_sorted);
i = find(cut>0);
if isempty(i)
  cut = 0;
else
  cut = cut(i(end));
end
w = y_weighted(nVal+1:end)>=cut;
% compute performance
tp = sum(w(1+nTest:end)==0);
fp = sum(w(1:nTest) == 0);
fn = sum(w(1+nTest:end) == 1);
tn = sum(w(1:nTest) == 1);
tpr = tp/(tp + fn + eps);
fpr = fp/(fp + tn + eps);
f1 = 2*tp/(2*tp + fn + fp);
auc_labels = boolean([ones(nTest,1); zeros(nAnomaly,1)]);
auc = scoreAUC(auc_labels, y_weighted(1+nVal:end));