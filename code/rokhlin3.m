%%%%%%%%%%%%%%%%%%%
% This function computes the normalized Rokhlin distance up to three RVs.
%
% Rd(A,B) = ( H(A|B) + H(B|A) ) / sqrt( H(A) * H(B) )
%
% Rd(A,B,C) = ( H(A|B,C) + H(B|A,C) + H(C|A,B) + I(A B C) ) / (
% H(A)*H(B)*H(C)) ^(1/3)
%
% Use: rd = rokhlin(X)
%
% Input:
%   X:    data matrix of size NxP, where N is the number of instances and P
%   is the numner of dimensions. Data is assumed to be discrete.
%
% Output:
%   rd:   Rokhlin multivariate distance
%
% The distance is normalized according to:
% Strehl, A. "Cluster Ensembles ? A Knowledge Reuse Framework for
% Combining Multiple Partitions", 2002, JMLR.
%
% Marcelo Bacher, Dec. 2015
% mgcherba@gmail.com
%
function rd = rokhlin3(X, w)

% check data dimensionality
P = size(X,2);
if P > 3
  % we do until second order...sorry
  disp('WARNING: Rokhlin Distance will be computed up to order 2!');
end

if nargin < 2
  w = 1;
end

if P > 2
  % a <- [H(X3) H(X2|X3) H(X1|X2,X3)]
  a = mvJointEntropy([X(:,3) X(:,2) X(:,1)]);
  % b <- [H(X3) H(X1|X3) H(X2|X1,X3)]
  b = mvJointEntropy([X(:,3) X(:,1) X(:,2)]);
  % c <- [H(X1) H(X2|X1) H(X3|X2,X1)]
  c = mvJointEntropy([X(:,1) X(:,2) X(:,3)]);
  % d <- I(X1;X2;X3) = I(X1;X2) - I(X1;X2|X3)
  d = mvMutualInformation(X(:,1), X(:,2)) - ...
    condMutualInformation(X(:,1), X(:,2), X(:,3));  
  % Rd = H(X1|X2,X3) + H(X2|X1,X3) + H(X3|X2,X1) + MI(X1 X2 X3)
  rd = a(3) + b(3) + c(3) + w*d;
  rd = rd / sum(abs(c));
elseif P == 2
  % we make it evident how to compute the Rokhlin distace
  % a <- [H(X2) H(X1|X2)]
  a = mvJointEntropy([X(:,2) X(:,1)]);
  % b <- [H(X1) H(X2|X1)]
  b = mvJointEntropy([X(:,1) X(:,2)]);
  % Rd = H(X1|X2) + H(X2|X1)
  rd = a(2) + b(2);
  rd =  rd / sum(abs(a));
else
  error('ERROR! X is a vector ...');
end