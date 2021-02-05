%===================================
% This function computes the Jaccard Index according to
% https://en.wikipedia.org/wiki/Jaccard_index
%
% Input:
%   A, B: set of indices
%
% Output:
%   ji:   Jaccard index
%
% Marcelo Bacher, Feb. 2016
% mgcherba@gmail.com
%
function ji = jaccard_index(A,B)
  G1unionG2 = union(A(:), B(:));
  G1intG2 = intersect(A(:), B(:));
  ji = length(G1intG2) / length(G1unionG2);