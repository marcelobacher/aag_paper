function [xd, bin] = discEqFreq2(x, n)

N = size(x,1);
xd = zeros(N,1);
bin = zeros(n,2);
xs = sort(x, 'ascend');
xs = unique(xs);
Nbin = length(xs);
for i=1:n
  k = floor((i-1)*Nbin/n)+1:floor(i*Nbin/n);
  try
  bin(i,1) = min(xs(k));
  catch e
    disp(e)
  end
  bin(i,2) = max(xs(k));
  j = (x >= bin(i,1)) & (x <= bin(i,2));
  xd(j>0) = .5*(bin(i,1)+bin(i,2));
end
