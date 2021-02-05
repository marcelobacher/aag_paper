function G = ewkm_subspace(xTrain)

% EWKM
P = size(xTrain,2);
K_range = 1:20;
Y = -[1/4 1/3 2/3];
[Kopt, distortion, transf_d] = get_best_nclusters(xTrain, ...
  K_range, Y);
Kopt = max(Kopt);
[Z, W, L, exit_cond] = ewkm(xTrain, min(P-2,Kopt), 1000);
% select subspaces in each cluster
a = cell(1,1);
[~,j] = max(L);
for k=1:Kopt
  b = find(j==k);
  if length(b) > 1
    a{k} = b;
  end
end
G = a;