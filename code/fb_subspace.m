function G = fb_subspace(xTrain)
% According to:
% Lazarevic, A. and V. Kumar, Feature bagging for outlier detection,
% KDD/05, 2005
P = size(xTrain,2);
T = 20;
G = cell(1,T);
for t=1:T
  % randomly chose the subspace size
  Ni = randsample(round(P/2):P-1,1);
  % randomly select Ni features
  G{1,t} = randsample(P,Ni);
end