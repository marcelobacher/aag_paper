function G = enclus_subspaces(xTrainDisc, f_enclus1, f_enclus2)
P = size(xTrainDisc,2);
H = zeros(P,1);
for i=1:P
  H(i) = mvJointEntropy(xTrainDisc(:,i));
end
Smax = 6;
b = zeros(1000,1);
for i=1:1000
  b(i) = sum(H(randsample(length(H),Smax)));
end
w = mean(b);
e = std(b);

% ENLCUS_SIG
S = enclus(xTrainDisc, w, f_enclus1*e, f_enclus2*e, Smax, 0, 1);

% reordering subspaces into one list
G = cell(1);
k = 1;
for i=1:length(S)
  g = S{i};
  for j=1:size(g,1)
    G{k} = g(j,:);
    k = k + 1;
  end
end
