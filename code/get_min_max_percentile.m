function [max_vect, min_vect] = get_min_max_percentile(X, p, minNumSym)
[N,dataDim] = size(X);
max_vect = zeros(1, dataDim);
min_vect = zeros(1, dataDim);
if (p>1.0) || (p<0)
  p = 0.90;
end
flag_nan = isnan(X);
for iDim = 1:dataDim
  dataValid = X(flag_nan(:,iDim)==0,iDim);
  dataSize = size(dataValid,1);
  if length(unique(dataValid)) > minNumSym
    s = sort(dataValid, 'ascend');
    max_vect(iDim) = s(round(p*dataSize));
    min_vect(iDim) = s(round((1-p)*dataSize)+1);
  elseif size(dataValid,1) > round(0.40*N)
    max_vect(iDim) = max(dataValid);
    min_vect(iDim) = min(dataValid);
  end
end
