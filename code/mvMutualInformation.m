%
% This function computes the multivariate mutual information between the
% matrix X and the variable y. X is a matrix whose dimension is NxP, where
% N is the number of instances and P the number of features. The vector y
% has the dimensionality Nx1
%
function MI = mvMutualInformation(X, y, verbose)

if nargin < 3
  verbose = 0;
end

N = size(X, 1);
if N ~= length(y)
  error('###ERROR!! length of y differs from instances in X!');
end

% compute joint entropy of featutes
if verbose == 1
  disp('Computing joint entropy of X ...');
end
% a = H(X1) + H(X2|X1) + H(X3|X2,X1) + H(X4|X3,X2,X1) + ...
a = mvJointEntropy(X);

% compute joint entropy of class and features
if verbose == 1
  disp('Computing joint entropy of y and X ...');
end
% b = H(Y) + H(X1|Y) + H(X2|X1,Y) + H(X3|X2,X1,Y) + ...
b = mvJointEntropy([y X]);

% this comes from MI(X,y) = sum H(Xi|Xi-1...X1) - sum H(Xi|Xi-1...X1, y)
% H(X1) + H(X2|X1) + H(X3|X2,X1) + ... - H(X1|Y) - H(X2|X1,Y) -
% H(X3|X2,X1,Y) ...
%
% the first entry in b is H(y) and therefore it is discarded in the sum
% below
MI = a - b(2:end);

