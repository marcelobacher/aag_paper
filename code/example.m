% example of agglomerative feature grouping
f = load('../data/pendigits-orig/pendigits_clean.mat');
fout = '../results/pendigits_performance.mat';

% set default random generator
rng('default');

% flag to add noise (1st setting)
add_noise = 0;
noise_factor = [.01 .03 .05 .07 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0];
n_noise_factor = max(1,length(noise_factor)*add_noise);

% confidence for anomaly detection in mv-set
alpha = 0.05;

%% get majority classes
classes = unique(f.labels);
if length(classes) > 2
  j = histc(f.labels, classes);
  [~,j] = sort(j, 'descend');
  targetClass = classes(j(1));
else
  % 1-0 ODDS dataset type
  targetClass = 0;
end
xTrain = f.data(targetClass == f.labels,:);
xAnomaly = f.data(targetClass ~= f.labels,:);
yAnomaly = f.labels(f.labels ~= targetClass);
prop_anomaly = size(xAnomaly,1) / size(xTrain,1);
if size(xAnomaly,1) > 0.20
  [~, ~, xAnomaly, ~] = splitTrainingData(xAnomaly, yAnomaly, .95);
end
% % if 0.1*size(xTrain,1) > 10
% %   [~, ~, xAnomaly, ~] = splitTrainingData(xAnomaly, yAnomaly, ...
% %     1-size(xTrain,1)*0.1/size(xAnomaly,1));
% % elseif size(xAnomaly,1) > 20
% %   [~, ~, xAnomaly, ~] = splitTrainingData(xAnomaly, yAnomaly, .95);
% % end
[nTrain, P] = size(xTrain);
% it improves the performance for anomaly detection approach
xTrain = xTrain .* (1 + 0.1*rand(nTrain, P));

% benchmarks
methods = {'aag'; 'fb'; 'enclus'; 'ewkm'; 'afg'; 'iforest'};
% methods = {'fb'};
% methods = {'enclus'};
% methods = {'ewkm'};
% methods = {'afg'};
% methods = {'iforest'};

%% main loop
nRep = 20;
performance = cell(length(methods),1);
perf_struct = struct('f1', 0, ...
  'auc', 0, 'mvmodel', [], 'pca', [], 'subspaces', [], 'w', []);
for i=1:length(methods)
  performance{i}.method = '';
  performance{i}.noise = add_noise;
  performance{i}.noise_factor = noise_factor;
  performance{i}.results = repmat(perf_struct, nRep, 1);
end

for r = 1:nRep
  % if data is too big, take coreset
  if nTrain > 2000
    [~, idx] = get_index_coreset(xTrain, 0.95, 100, 50);
    idx_diff = setdiff((1:nTrain)', idx);
     xTestAppend = xTrain(idx_diff,:);
     xTrain = xTrain(idx,:);
  else
    xTestAppend = [];
  end
  nTrain = size(xTrain,1);
  % split of train, validation and test data sets
  if isempty(xTestAppend)
    % test dataset for ensemble of anomaly detection
    selection = zeros(nTrain,1);
    selection(randsample(nTrain, round(.7*nTrain))) = 1;
    xTest = xTrain(selection == 0, :);
    xTrain = xTrain(selection > 0, :);
    nTrain = size(xTrain,1);
  else
    xTest = xTestAppend;  
  end
  
  % validation dataset only for MV-set model
  if nTrain > 200
    selVal = zeros(nTrain,1);
    selVal(randsample(nTrain, round(.3*nTrain))) = 1;
    xTrainAn = xTrain(selVal == 0,:);
    xVal = xTrain(selVal>0,:);
    nVal = size(xVal,1);
  else
    xTrainAn = xTrain;
    xVal = xTrain;
    nVal = nTrain;
  end
  
  % remove nonvalid attributes
  valid_att = std(xTrain, 1) > 1e-4;
  xTrain = xTrain(:, valid_att);
  xTest = xTest(:, valid_att);
  
  % imputation of missing values using only training data
  xTrain = impute_mean(xTrain);
  mData = mean(xTrain);
  xTest = impute_mean(xTest, mData);
  nTest = size(xTest, 1);
  nTrain = size(xTrain, 1);
  
  % noise addition test
  if add_noise == 1
    n_noise = round(.5*nTest);
    selection_noise = randsample(nTest, n_noise);
    xAnomaly = xTest(selection_noise,:);
    xAnomaly_orig = xTest(selection_noise,:);
    xAnomaly_orig = impute_mean(xAnomaly_orig, mData);
  else
    xAnomaly = xAnomaly(:, valid_att);
    xAnomaly = impute_mean(xAnomaly, mData);
  end
  nAnomaly = size(xAnomaly,1);
    
  % selection of subspaces
  for m = 1:length(methods)
    performance{m}.method = methods{m};
    if strcmpi(methods{m}, 'aag')
      [xTrain_d, ~] = discretize_data(xTrain);
      [T, ~] = maag4(xTrain_d, [], 0, 0, 0, 1, 1, 1);
      % clean up identical groups for anomaly detection ensemble
      G = cleanup_groups(T);

    elseif strcmpi(methods{m}, 'fb')
      G = fb_subspace(xTrain);
      
    elseif strcmpi(methods{m}, 'enclus')
      [xTrain_d, ~] = discretize_data(xTrain);
      f_enclus1 = 0.5;
      f_enclus2 = 1.5;
      G = enclus_subspaces(xTrain_d, f_enclus1, f_enclus2);
      
    elseif strcmpi(methods{m}, 'ewkm')
      G = ewkm_subspace(xTrain);
      
    elseif strcmpi(methods{m}, 'afg')
      G = afg_k_means_subspace(xTrain);
      
    elseif strcmpi(methods{m}, 'iforest')
      NumTree = 100; % number of isolation trees
      NumSub = max(20, min(floor(nTrain/4),max(256, nTrain))); % subsample size
      NumDim = size(xTrain, 2); % do not perform dimension sampling
      iTree = IsolationForest(xTrain, NumTree, NumSub, NumDim, ...
        randi(10000));
      auc = [];
      f1 = [];
      
      for k=1:n_noise_factor
        if add_noise == 1
          dim = randsample(P, min(P, round(noise_factor(k)*P+1)));
          noise_std = std(XoTest_orig(:,dim));
          
          xAnomaly(:, dim) = bsxfun(@plus, xAnomaly_orig(:,dim), ...
            bsxfun(@times, noise_std, mvnrnd(zeros(length(dim),1), ...
            eye(length(dim)), n_noise)));
        end
        [Mass, ~] = IsolationEstimation([xTest; xAnomaly], iTree);
        Score = mean(Mass, 2);
        labels = boolean([ones(nTest,1); zeros(nAnomaly,1)]);
        auc = [auc; scoreAUC(labels, Score)];
        index_cut = max(1, floor(alpha*length(labels)));
        [score_sorted, index_score] = sort(Score,'ascend');
        score_cut = score_sorted(index_cut);
        yhat = double(Score >= score_cut);
        tp = sum((yhat == 0) & (labels == 0));
        fp = sum((yhat == 0) & (labels == 1));
        fn = sum((yhat == 1) & (labels == 0));
        tn = sum((yhat == 1) & (labels == 1));
        f1 = [f1; 2 * tp / (2 * tp + fp + fn)];
      end
    else
      error([methods{m} ' not implemented!!']);
    end
    
    % build anomaly detection models for ensemble
    if ~strcmpi(methods{m}, 'iforest')
      yhat = [];
      mvset_model = cell(1,1);
      model_pca = cell(1,1);
      f1 = [];
      auc = [];
      w = [];
      for g = 1:length(G)
        fs = G{g};
        [result, train_pdf, ~, model_pca{g}, mvset_model{g}, ...
          test_pdf, ~] = find_mv_set(xTrainAn(:,fs), ...
          [xVal(:,fs); xTest(:,fs); xAnomaly(:,fs)], 0.9, alpha);
        if isempty(mvset_model{g})
          warning('MV-set model could not be computed!');
        else
          if add_noise == 0
            yhat = [yhat result];
          end
        end
      end
      
      for k=1:n_noise_factor
        if add_noise == 1
          dim = randsample(P, min(P, round(noise_factor(k)*P+1)));
          noise_std = std(xAnomaly_orig(:,dim));
          
          xAnomaly(:, dim) = bsxfun(@plus, xAnomaly_orig(:,dim), ...
            bsxfun(@times, noise_std, mvnrnd(zeros(length(dim),1), ...
            eye(length(dim)), n_noise)));
          
          yhat = [];
          for g=1:length(G)
            fs = G{g};
            if ~isempty(mvset_model{g})
              x = [xVal(:,fs); xTest(:,fs); xAnomaly(:,fs)];
              x = bsxfun(@minus, x, mData(fs));
              x_train = bsxfun(@minus, xTrain(:,fs), ...
                mData(fs));
              if ~isempty(model_pca{g})
                x = x * model_pca{g};
                x_train = x_train * model_pca{g};
              end
              x_pdf = mvset_model{g}.pdf(x);
              train_pdf = mvset_model{g}.pdf(x_train);
              [train_pdf, ~] = sort(train_pdf);
              cut_mv_set = floor(nTrain*(1-alpha));
              result = double(x_pdf >= train_pdf(nTrain-cut_mv_set));
              yhat = [yhat result];
            end
          end
          [f1_noise, auc_noise, w_noise, ~] = calc_performance(yhat, ...
            [ones(nVal+nTest,1); ones(nAnomaly,1)], nVal, nTest, nAnomaly);
          f1 = [f1 f1_noise];
          auc = [auc auc_noise];
          w = [w w_noise];
        else
          [f1, auc, w, ~, ~] = calc_performance(yhat, ...
            [ones(nVal+nTest,1); ones(nAnomaly,1)], nVal, nTest, nAnomaly);
        end
      end
    end
    % record results    
    performance{m}.results(r).f1 = f1;
    performance{m}.results(r).auc = auc;
    if ~strcmpi(methods{m}, 'iforest')
      performance{m}.results(r).mvmodel = mvset_model;
      performance{m}.results(r).pca = model_pca;
      performance{m}.results(r).subspaces = G;
      performance{m}.results(r).w = w;
    else
      performance{m}.results(r).mvmodel = iTree;      
    end
  end
end

% record results
save(fout, 'performance');









