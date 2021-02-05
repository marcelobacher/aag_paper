function [Kopt, distortion, transf_d] = get_best_nclusters(data, ...
  K_range, Y, param, verbose)

if nargin < 5
  verbose = 0;
end

if nargin < 4
  verbose = 0;
  param.method = 'kmeans';
end
  
if nargin < 3
  error('ERROR get_best_nclusters!! Missing input parameters!!');
end

if isfield(param, 'method')
  if strcmp(param.method, 'fsc')
    if ~isfield(param, 'alpha')
      error('ERROR FSC !! alpha field is not defined in param!!');
    end
  elseif strcmp(param.method, 'ewkm')
    if ~isfield(param, 'gamma')
      error('ERROR EWKM !! gamma field is not defined in param!!');
    end    
  elseif strcmp(param.method, 'mssc')
    if ~isfield(param, 'alpha')
      error('ERROR MSSC !! alpha field is not defined in param!!');
    end
    if ~isfield(param, 'beta')
      error('ERROR MSCC !! beta field is not defined in param!!');
    end            
  end
else
  error(['ERROR get_best_nclusters!! method '...
    'field is not defined in param!!']);
end

[n,p] = size(data);

% first entry in transformed
K = max(K_range);
transf_d = zeros(K+1,length(Y));

% distortion
distortion = zeros(K,1);

% main loop
for k = 1:min(K, n-2)
  if verbose > 0
    disp(['Clustering with K = ' num2str(k) ' ...']);
  end
  % fuzzy subspace clustering
  if strcmp(param.method, 'fsc')
    tol = 1e-5;
    alpha = 2;
    [Z, W, U, ~] = fsc(data, k, param.alpha);
    Z = Z';
    W = W';
  elseif strcmp(param.method, 'kmeans')
    % regular k-means
    [U, Z, sumD] = kmeans(data, k, 'emptyaction', 'singleton');
    
  elseif strcmp(param.method, 'ewkm')
    gamma = 1;
    [Z, U, W] = ewkm(data, k, param.gamma);
    W = W';
    m = isnan(W);
    W(m) = 1;
    Z=Z';
    
  elseif strcmp(param.method, 'mssc')
    [W, Z, U, ~] = mssc(data, param.alpha, param.beta, k, 0);
    U = U';
  end
  
  % for all other clusterings except ...
  if 0 == strcmp(param.method, 'kmeans') && ...
     0 == strcmp(param.method, 'mssc') 
    for i=2:k
      l = U(:,i)==1;
      U(l,i) = i;
    end
    U = sum(U,2);
  end
  
  % compute distortion
  b = zeros(k,1);
  
  for i=1:k
    if strcmp(param.method, 'kmeans')
        a = sumD(i);
    
    elseif strcmp(param.method, 'fsc')
      a = sum(sum(bsxfun(@times, ...
      bsxfun(@minus, data(U==i,:), Z(i,:)).^2, (W(i,:).^(param.alpha))),2));% ...
      %+tol*sum(W(i,:).^alpha);
             
    elseif strcmp(param.method, 'ewkm')
      a = sum(sum(bsxfun(@times, ...
      bsxfun(@minus, data(U==i,:), Z(i, :)).^2, W(i,:)),2)); %+ ...
      %-gamma*sum(W(i,:).*log(W(i,:)+eps));
    
    elseif strcmp(param.method, 'mssc')
      a = sum(sum(bsxfun(@times, ...
      bsxfun(@minus, data(U==i,:), Z(i,:)).^2, (W(i,:).^(param.alpha))),2)); %+ ...
      %+tol*sum(W(i,:).^alpha);    
    end
    b(i) = a;
  end
  
  distortion(k) = sum(b)/(n*p);
  for y=1:length(Y)
    transf_d(k+1, y) = distortion(k).^(Y(y));
  end
end

% return #of clusters according to maximal transformed distortion rate
[~, Kopt] = max(diff(transf_d));
Kopt = Kopt + 1;
