function G = afg_k_means_subspace(xTrain)
P = size(xTrain,2);
K_range = 1:20;
Y = -[1/4 1/3 2/3];
[Kopt, distortion, transf_d] = get_best_nclusters(xTrain, K_range, Y);
Kopt = max(Kopt);
[U, Z, W, G, V, L, exit_cond] = afg_km(xTrain, Kopt, min(P-2,Kopt));
a = cell(1,1);
kg = 1;
for k=1:size(G,2)
  if sum(G(:,k))> 1
    a{kg} = find(G(:,k)>0);
    kg = kg + 1;
  end
end
G = a;