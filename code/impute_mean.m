function data_out = impute_mean(data_in, M)

P = size(data_in,2);

% check missing data after scaling
if nargin < 2
  M = nanmean(data_in);
end

data_out = data_in;
k = isnan(data_in);
if sum(k(:)) > 0
  for n=1:P
    data_out(k(:,n)~=0,n) = M(n);
  end
end

