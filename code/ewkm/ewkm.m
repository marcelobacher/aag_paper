%%
% [Z, W, L, exit_cond] = ewkm(X, K, gamma, e, nIter)
%
% This function is the implementation of Entropy Weighted K-Means algorithm
% presented in the paper:
%
% Liping Jing, Michael K. Ng, and Joshua Zhexue Huang, An Entropy Weighting
% k-Means Algorithm for Subspace Clustering of High-Dimensional Sparse
% Data, IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING,	VOL. 19,	NO.
% 8,	AUGUST 2007
%
% Input:
%   X:    data matrix of dimension n x p, where n is the number of
%   instances and p the data dimensionality (obligatory)
%   K:    number of clusters (obligatory)
%   gamma:  gamma parameter (optional, default 1)
%   e:    exit condition for objective function (optional default 1e-3)
%   nIter:  number of maximal iterations (optional, default 1000)
%
% Output:
%   Z:    centroid matrix of dimension p x K
%   W:    indication matrix of each instance of dimension n x K
%   L:    lambda matrix of dimension p x K
%   exit_cond:  array with objective function values
%
% Marcelo Bacher, January 2016
%===============================
function [Z, W, L, exit_cond] = ewkm(X, K, gamma, e, nIter)

% check arguments
if nargin == 4
  nIter = 1000;
end

if nargin == 3
  nIter = 1000;
  e = 1e-3;
end

if nargin == 2
  nIter = 100;
  e = 1e-3;  
  gamma = 1;
end

if nargin < 2
  error('EWKM ERROR!! Insufficient number of arguments!');
end

% get data dimesnionality
[n, m] = size(X);

% initialize algorithms based on k-means
[~, Z] = kmeans(X, K, 'emptyaction', 'singleton');
%Z = X(randsample(n,K),:)';

% initialize weights matrices
L = (1/m)*ones(K, m);

% convergence criterion
exit_cond = zeros(nIter,1);

for iter = 1:nIter
  % update assignment matrix W
  a = zeros(n, K);
  W = zeros(n, K);
  for k=1:K
    % compute weighted distance to centroids
    a(:,k) = sum(bsxfun(@times, ...
      bsxfun(@minus, X, Z(k,:)).^2, L(k,:)), 2);
  end
  
  [~, j] = min(a,[],2);
  for i=1:n
    W(i,j(i)) = 1;
  end
  
  
  % update centroid matrix Z
  for k=1:K
    Z(k,:) = sum(X(W(:,k)>0,:),1)/sum(W(:,k));
  end
  
  % update weight matrix L
  for k=1:K
    D = sum(bsxfun(@minus, X(W(:,k)>0,:), Z(k,:)).^2, 1);
    L(k,:) = exp(-D/gamma)/sum(exp(-D/gamma));
  end
  
  exit_cond(iter) = calc_objective(X, Z, L, W, K, gamma);
  if iter > 1
    % compute objective function and update exit condition
    if abs(exit_cond(iter-1)-exit_cond(iter)) <= e
      break;
    end
  end
end


function o = calc_objective(X, Z, L, W, K, gamma)

  J = zeros(K,1);
  for k=1:K
    J(k) = sum(sum(bsxfun(@times, bsxfun(@minus, X(W(:,k)>0,:), ...
      Z(k,:)).^2, L(k,:)),2)) + gamma*sum(L(k,:).*log(L(k,:)+eps));
  end
  
  o = sum(J);



