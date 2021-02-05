%%
% [U, Z, W, G, V, L, exit_cond] = afg_km(X, K, T, beta, e1, e2, Nmax, delta)
%
% This function is the implementation of Automatic Feature Grouping K-Means
% referred as AFG-Kmeans presented in the paper:
%
% Gan, G. and Kwok-Po Ng, M., (2015),  Subspace clustering with automatic feature
% grouping, Pattern Recognition 48, 3703?3713, 
%
% Input:
%   X:    data matrix of dimension n x m, where n is the number of
%   instances and m the data dimensionality (obligatory)
%   K:    number of clusters (obligatory)
%   T:  number of groups (obligatory)
%   beta:  beta parameter (optional, default 1)
%   e1:    positive constant (optional default 1e-3)
%   e2:   positive constant (optional default 1e-3)
%   Nmax:  number of maximal iterations (optional, default 100)
%
% Output:
%   U:    indication matrix of data (n x K)
%   Z:    centroid matrix of dimension m x K
%   W:    feature weight matrix K x m
%   G:    feature group matrix K x T
%   V:    group weight matrix K x T
%   L:    lambda matrix of dimension K x T
%   exit_cond:  array with objective function values
%
% Marcelo Bacher, January 2017
%===============================
function [U, Z, W, G, V, L, exit_cond] = ...
  afg_km(X, K, T, beta, e1, e2, Nmax, delta)

% check parameters
if nargin < 3
  error('AFG_KM: Error!! not enought number of obligatory parameters!');
elseif nargin < 4
  beta = 1;
  e1 = 1e-4;
  e2 = 1e-4;
  Nmax = 100;
  delta = 1e-5;
elseif nargin < 5;
  e1 = 1e-4;
  e2 = 1e-4;
  Nmax = 100;
  delta = 1e-5;
elseif nargin < 6
  e2 = 1e-4;
  Nmax = 100;
  delta = 1e-5;
elseif nargin < 7
  Nmax = 100;
  delta = 1e-5;
elseif nargin < 8
  delta = 1e-5;
end  

% get data dimesnionality
[n, m] = size(X);

% initialization of assignment matrix
U = ones(n, K);

% initialize algorithms based on k-means
[IDX, Z] = kmeans(X, K, 'emptyaction', 'singleton');

% based on kmeans, intialize matrix U
for k=1:K
  U(:, k) = double(IDX==k);
end

% initialize group weights matriX
V = ones(K, T);

% initialization of feature membership matrix
G = zeros(m, T);
G(:,1) = 1;

% initialization of featute weights matrix
W = ones(K, m);

% initialization of group weights
L = ones(K, T);

% convergence criterion
exit_cond = 10*delta*ones(Nmax,1);

for iter = 1:Nmax
  
  % update centroid matrix Z
  for k=1:K
    Z(k,:) = sum(X(U(:,k)>0,:),1)/sum(U(:,k));
  end
  
  % compute weighted distance
  a = zeros(n, K);
  for k=1:K
    a(:,k) = sum(bsxfun(@times, bsxfun(@minus, X, ...
      Z(k,:)).^2, W(k,:).^2),2);
  end
  
  % update matrix U
  [~, j] = min(a,[],2);
  for i=1:n
    U(i,j(i)) = 1;
  end
  
  % update matrix W
  for k=1:K
    E = e1 + sum(bsxfun(@minus, X(U(:,k)>0,:), Z(k,:)).^2, 1);
    a = 0;
    b = 0;
    for t=1:T
      a = a + G(:,t)*L(k,t)^2;
      b = b + G(:,t)*V(k,t)*L(k,t)^2;
    end
    term1 = beta*a(:) + E(:);    
    term2 = sum(1./(term1 + eps));    
    term3 = 2*sum(beta*b(:)./term1(:));
    lambda = (-2.*m + term3)/term2;
    
    W(k,:) = ((beta*b(:) - 0.5*lambda)./term1(:))';
  end  
  
  if iter == 1
    col = randsample(m, T);
    V = W(:,col);
  else
    % update V matrix
    for k=1:K
      for t=1:T
        V(k,t) = sum(W(k,G(:,t)>0))/sum(G(:,t));
      end
    end
  end
  
  % update matrix G
  a = zeros(m, T);
  for t=1:T
    a(:,t) = sum(bsxfun(@times, bsxfun(@minus, W, V(:,t)).^2, L(:,t)),1)';
  end
  [~, j] = min(a,[],2);
  G = zeros(m, T);
  for i=1:m
    G(i,j(i)) = 1;
  end
  
  % compute objective and check for termination
  exit_cond(iter) = calc_objective(U, Z, W, G, V, L, X, beta, e1, e2);
  if iter > 1;
    if abs(exit_cond(iter-1)-exit_cond(iter)) <= delta
      break;
    end
  end
end


function Q = calc_objective(U, Z, W, G, V, L, X, beta, e1, e2)

K = size(W,1);
T = size(V,2);

a = zeros(K,1);
for k=1:K
  a(k) = sum(sum(bsxfun(@times, bsxfun(@minus, X(U(:,k)>0,:), Z(k,:)).^2, W(k,:)),2));
  a(k) = a(k) + e1 * sum(W(k,:).^2);
end

b = zeros(T,1);
for t=1:T
  b(t) = sum(sum(bsxfun(@times, bsxfun(@minus, W(:,G(:,t)>0), V(:,t)).^2, L(:,t)),2));
  b(t) = b(t) + e2*sum(L(:,t).^2);
end

Q = sum(a) + beta*sum(b);


