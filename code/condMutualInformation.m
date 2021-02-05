function condmi = condMutualInformation(y,x,z)
  % a = H(z) + H(y|z)
  a = mvJointEntropy([z y]);
  
  % b = H(x) + H(z|x) + H(y|x,z)
  b = mvJointEntropy([x z y]);
  
  % I(y;x|z) = H(y|z) - H(y|x,z)
  condmi = a(2) - b(3);
end