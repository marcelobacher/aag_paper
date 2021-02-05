function mmi = multivariate_mi(X)

p = size(X,2);
mmi = 0;
for i=1:p
  k = nchoosek(1:p,i);
  t = (-1)^i;
  for j=1:size(k,1)
    mmi = mmi + t*sum(mvJointEntropy(X(:,k(j,:))));
  end
end
mmi = -1 * mmi;
