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

if nargin < 2
  w = 1;
end

rd = rokhlin_n(X, 1);
