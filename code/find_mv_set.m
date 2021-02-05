function [result, ePdf, Ay, B, model, test_pdf, type_model] = find_mv_set(xTrain, xTest, p, alpha)
% Method according to:
% Park, C., et al, A Computable Plug-In Estimator of Minimum
% Volume Sets for Novelty Detection, INFORMS, 2010
result = [];
ePdf = [];
Ay = [];
B = [];
model = [];
test_pdf = [];

mData = mean(xTrain);
L = bsxfun(@minus, xTrain, mData);
L_test = bsxfun(@minus, xTest, mData);
nTrain = size(xTrain,1);
nTest = size(xTest,1);
% pre-process via PCA
if size(xTrain,2)>3
  [~, B] = doPca(xTrain, p, 1, 1);
  % compute projection
  L = L*B;
  L_test = L_test*B;
else
  B = [];
end

% estimate distribution
% if size(L,2) == 2
%   type_model = 1;
%   % KDE2
%   d = 2;
%   s = .5 * (d + 3)/((d+2)*d+4) + .5 * (2*d + 3)/(2*(2 + d)^2);
%   % h = nTrain^(-s);
%   kde2.h = [s; s];
%   kde2.N = 200;
%   model = gkde2(L, kde2);
%   ePdf = zeros(nTrain,1);
%   % empirical probability
%   for i=1:nTrain
%     j = find(L(i,1)<model.x(1,:));
%     q = find(L(i,2)<model.y(:,1));
%     ePdf(i) = model.pdf(j(1), q(1));
%   end
%   [s_ePdf, index] = sort(ePdf);
%   % compute the minimal volume set using alpha
%   j = floor(nTrain*(1-alpha));
%   Ay = L(index(1:j),:);
%   % test data for subspace indexed by fs
%   %nn = size(L_test,1);
%   test_pdf = zeros(nTest,1);
%   for i=1:nTest
%     w = find(L_test(i,1)<model.x(1,:));
%     q = find(L_test(i,2)<model.y(:,1));
%     if ~isempty(w) && ~isempty(q)
%       test_pdf(i) = model.pdf(w(1), q(1));
%     end
%   end
% else
  type_model = 2;
  % GMM
  minAIC = Inf;
  model = [];
  % maxM = min(size(xTrain,1), size(xTrain,2));
  % maxM = maxM - 1;
  maxM = min(5, min(size(xTrain,1), size(xTrain,2)));
  if size(xTrain,1) <= size(xTrain,2)
      reg = 1;
  else
      reg = 0.1;
  end
  % Xi must have more rows than the number of components.
  for m=1:maxM
    try
      aux = gmdistribution.fit(L, m, 'Regularize', reg);
      if minAIC > aux.AIC;
        model = aux;
        minAIC = aux.AIC;
      end
    catch e
      disp(e)
    end
  end
  % compute empirical probabilities
  if ~isempty(model)
    ePdf = model.pdf(L);
    if size(ePdf(:)) ~= size(L,1)
      disp('Puto');
    end
    [s_ePdf, index] = sort(ePdf);
    % compute the minimal volume set using alpha(u)
    j = floor(nTrain*(1-alpha));
    Ay = L(index(1:index),:);
    % compute probabilities on test data
    test_pdf = model.pdf(L_test);
  end
% end

% resturn result
if ~isempty(model)
  result = double(test_pdf >= s_ePdf(nTrain-j));
end
