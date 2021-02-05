%%
% SU = mvSymmUnc(x,y)
%
% This function computes the symmetric uncertainity as the coscient between
% the mutual information between x and y and the sum of each entropy:
%
% SU = 2*I(X;Y) / (H(X) + H(Y))
% 
% a valud of 0 means complete independance whilst a value of 1 means
% complete dependance.
%
% The definition of SU is according to:
%
% Yu, L. and Liu, H., Efficient Feature Selection via Analysis of
% Relevance and Redundancy, JMLR, 2004
%
function su = mvSymmUnc(x,y)
hx = mvJointEntropy(x);
hy = mvJointEntropy(y);
mi = mvMutualInformation(x,y);
if (hx > eps) || (hy > eps)
  su = 2*mi/(hx + hy + eps);
else
  su = 0;
end
