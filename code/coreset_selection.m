function [C, idx] = coreset_selection(X, p)
indices = (1:size(X,1))';
% sample uniformly the first centroid candidate
sampled = randsample(size(X,1),1);
C = X(sampled, :);
idx = sampled;
% remove selected candidate centroid from data
X(sampled, :) = [];
indices(sampled) = [];
% compute cost of candiates in C
[psi0, d] = l_calc_psi(C, X);
px = p * d / psi0;
% select further centroid candidates
for i = 1:round(log(psi0))
  % sample x in X w.p. px
  u = rand(size(X,1), 1);
  j = find(u < px);
  if ~isempty(j)
    % update list of sampled candidates
    C = [C; X(j,:)];
% %     idx = [idx; j];
    idx = [idx; indices(j)];
    % remove selected candidates from data
    X(j,:) = [];
    indices(j) = [];
    [psi,d] = l_calc_psi(C, X);
    % it returns the prob. for each x not selected candidate.
    px = p * d / psi;
  end
end

function [psi,d] = l_calc_psi(C, X)
[~, d] = l_calc_dist(C, X);
psi = sum(d);

function [L, d] = l_calc_dist(C, X)
[d,L] = max(bsxfun(@minus, 2.0*real(X*C'), dot(C, C, 2).'), [], 2);