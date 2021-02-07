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
function d = rokhlin_n(X, normalized)

if nargin < 2
  normalized = 0;
end
% check data dimensionality
P = size(X,2);
d = 0;
H = sum(abs(mvJointEntropy(X)));
if P == 3
  mi = zeros(P,2);
end
for i=1:P
  indices = 1:P;
  indices(i) = [];
% %   a = mvJointEntropy([X(:,indices) X(:,i)]);
  a = mvJointEntropy(X(:,indices));
  if P == 3
    mi(i,:) = a(:)';
  end
  d = d + H - sum(abs(a));
end
if mod(P,2)>0
  if P > 3
    d = d + multivariate_mi(X);
  elseif P == 3
    d = d + mi(1,1) + mi(2,1) + mvJointEntropy(X(:,3)) - sum(mi(:)) + H;
  end
end
if (normalized) > 0 && (H > 0)
  d = d / H;
end

