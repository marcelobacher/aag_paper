function H = mvJointEntropy(X, verbose)

if nargin < 2
  verbose = 0;
end

% get data dimension
P = size(X, 2);
% return vector
H = zeros(P, 1);

% iterate over each dimension
for p=1:P
  if verbose > 0
    disp([' Getting feature: ' num2str(p) ' out of ' num2str(P) '...']);
  end
  % compute recursive H(X1, X2,...,XP) = sum H(Xi | Xi-1, Xi-2,...,X1)
  H(p) = localCondEntropy(X(:,1:p));
end

% this local function computes recursively conditional entropy on X
function h = localCondEntropy(X)

% get data dimension
[N, P] = size(X);
h = 0;
if P==1
  partitions = unique(X);
  px = histc(X, partitions)/N;
  h = -sum(px.*log2(px+eps));
  if h <= eps
    h = 0.0;
  end
else
  partitions = unique(X(:,1));
  px = histc(X(:,1), partitions)/N;
  for k=1:length(partitions)
    u = X(:,1) == partitions(k);
    h = h + px(k)*localCondEntropy(X(u,2:P));
  end
end
