function [A,P] = doPca(data, percent, doMean, doSvd)

if nargin == 1
    percent = .9;
    doMean = 0;
    doSvd = 0;
elseif nargin == 2
    doMean = 0;
    doSvd = 0;
elseif nargin == 3
  doSvd = 0;
end

[n,p] = size(data);

if doMean
    M = mean(data);
else
    M = zeros(1,p);    
end

if doSvd == 0
  S = cov(data - ones(n,1)*M);
  [U,V] = eig(S);
else
  [U,V,~] = svd(data');
  V = V*V';
end
V = diag(V);
[~, index] = sort(-1*V);
V = V(index);
U = U(:,index);
% get principal components
limit = find((cumsum(V)/sum(V))<=percent);
if isempty(limit)
  A = V(1:2);
  P = U(:, 1:2);
else
  A = V(limit);
  P = U(:, limit);
end

