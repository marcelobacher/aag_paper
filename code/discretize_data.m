function [xTrain_d, disc] = discretize_data(xTrain)
% discretization of training attributes
P = size(xTrain, 2);
disc = zeros(P,1);
xTrain_d = xTrain;
bin = cell(P,1);
maxSymbols = 50;
for i = 1:P
  singleVal = unique(xTrain(:,i));
  if length(singleVal) > maxSymbols
% %     disc(i) = calcnbins(singleVal, 'fd', 1, maxSymbols);
    disc(i) = maxSymbols;
    [xTrain_d(:,i), bin{i}] = discEqFreq2(xTrain(:,i), disc(i));
  else
    disc(i) = length(unique(xTrain(:,i)));
  end
end