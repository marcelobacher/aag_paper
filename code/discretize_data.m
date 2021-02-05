function [xTrain_d, disc] = discretize_data(xTrain)
% discretization of training attributes
P = size(xTrain, 2);
discret.k = zeros(P,1);
disc = zeros(P,1);
xTrain_d = xTrain;
bin = cell(P,1);
for i = 1:P
  singleVal = unique(xTrain(:,i));
  if length(singleVal) >= 100
    disc(i) = calcnbins(singleVal, 'fd', 1, 50);
    if discret.k(i) >= 50
      [xTrain_d(:,i), bin{i}] = discEqFreq2(xTrain(:,i), disc(i));
    end
  else
    disc(i) = length(unique(xTrain(:,i)));
  end
end