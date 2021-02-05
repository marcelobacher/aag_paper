function results = calcRankMethod(data, indexBase)
[N,K] = size(data);
[~, position] = sort(data, 2, 'descend');
indexData = zeros(N,K);
for i=1:N
  indexData(i, position(i,:)) = 1:K;
end
avgRank = mean(indexData,1);
avgRank2 = avgRank.^2;
% avgRankAll = mean(avgRank);
sumAvgRnk2 = sum(avgRank2);
k1 = 0.25*K*(K+1)^2;
k2 = 12*N / (K*(K+1));
friedman = k2*(sumAvgRnk2-k1);
% p_chi = [9.213 5.992];
% p_F = [5.211 3.425];
imanRap = (N- 1)*friedman/(N*(K-1) - friedman);
selIndex = ones(K,1);
selIndex(indexBase) = 0;
diffRank = abs(avgRank(indexBase) - avgRank(selIndex>0));
stdErr = sqrt(K*(K+1)/(6*N));
zStat = diffRank/stdErr;
pValue = normpdf(zStat, 0, stdErr);

results.indexData = indexData;
results.diffRank = diffRank;
results.zStat = zStat;
results.pValue = pValue;
results.stdErr = stdErr;
results.friedman = friedman;
results.imanRap = imanRap;

