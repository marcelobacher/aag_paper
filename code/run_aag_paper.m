function performance = run_aag_paper(data, labels, methods, alpha, ...
  add_noise, noise_factors, nRep, nRepVal)

% set default random generator
rng('default');

% flag to add noise (1st setting)
n_noise_factor = max(1,length(noise_factors)*add_noise);

%% get majority classes
classes = unique(labels);
j = histc(labels, classes);
[~,j] = sort(j, 'descend');
targetClass = classes(j(1));
xTrain = data(targetClass == labels,:);
xAnomaly = data(targetClass ~= labels,:);
yAnomaly = labels(labels ~= targetClass);
prop_anomaly = size(xAnomaly,1) / size(xTrain,1);
% % if prop_anomaly > 0.20
% %   idx = find(labels ~= targetClass);
% %   [~, ~, xAnomaly, ~, ~, i_anomaly] = splitTrainingData(xAnomaly, yAnomaly, .95);
% %   i_anomaly = idx(i_anomaly);
% % else
% %   i_anomaly = find(labels ~= targetClass);
% % end
[nTrain, P] = size(xTrain);
% it improves the performance for anomaly detection approach
xTrain = xTrain .* (1 + 0.1*rand(nTrain, P));

%% main loop
performance = cell(length(methods),1);
perf_struct = struct('f1', 0, ...
  'auc', 0, 'mvmodel', [], 'pca', [], 'subspaces', [], ...
  'subspaces_weights', [], 'w', [], ...
  'train', [], 'val', [], 'test', [], 'anomaly', []);
for i=1:length(methods)
  performance{i}.method = '';
  performance{i}.noise = add_noise;
  performance{i}.target = targetClass;
  performance{i}.noise_factors = noise_factors;
  performance{i}.results = repmat(perf_struct, nRep, 1);
end

% impute missing values
xTrain = impute_mean(xTrain);
mData = mean(xTrain);

fprintf('### Starting main loop....\n');
for r = 1:nRep
  fprintf('##### Starting repetition %d / %d....\n', r, nRep);
  
  % select anomalies
  if prop_anomaly > 0.20
    idx = find(labels ~= targetClass);
    [~, ~, xAnomalyRep, ~, ~, i_anomaly] = splitTrainingData(xAnomaly, yAnomaly, .95);
    i_anomaly = idx(i_anomaly);
  else
    i_anomaly = find(labels ~= targetClass);
  end
  
  % if data is too big, take coreset  
  if nTrain > 2000
    fprintf('##### Reducing data with coreset computation....\n');
    idx = get_index_coreset(xTrain, 0.95, 100, 50);
    idx_diff = setdiff((1:nTrain)', idx);
    xTestAppend = xTrain(idx_diff,:);
    xTrainRep = xTrain(idx,:);
    i_train = idx;
    nTrain = size(xTrainRep,1);
  else
    xTestAppend = [];
    nTrain = size(xTrain,1);
  end
  
  % split of train, validation and test data sets
  if isempty(xTestAppend)
    % test dataset for ensemble of anomaly detection
    selection = zeros(nTrain,1);
    selection(randsample(nTrain, round(.7*nTrain))) = 1;
    xTestRep = xTrain(selection == 0, :);
    xTrainRep = xTrain(selection > 0, :);
    nTrain = size(xTrainRep,1);
    % record indices of selected data
    i_train = find(selection > 0);
    i_test = find(selection == 0);
  else
    xTestRep = xTestAppend;  
    i_test = idx_diff;
  end
  
  % validation loop
  nRepVal = max(1,double((nTrain > 200) * nRepVal));
  
  % remove nonvalid attributes
  valid_att = std(xTrainRep, 1) > 1e-4;
  xTrainRep = xTrainRep(:, valid_att);
  xTestRep = xTestRep(:, valid_att);
  
  % imputation of missing values using only training data
  xTestRep = impute_mean(xTestRep, mData(valid_att));
  nTest = size(xTestRep, 1);
  
  % noise addition test
  if add_noise == 1
    n_noise = round(.5*nTest);
    selection_noise = randsample(nTest, n_noise);
    xAnomalyRep = xTestRep(selection_noise,:);
    xAnomaly_orig = xTestRep(selection_noise,:);
    xAnomaly_orig = impute_mean(xAnomaly_orig, mData);
  else
    xAnomalyRep = xAnomalyRep(:, valid_att);
    xAnomalyRep = impute_mean(xAnomalyRep, mData(valid_att));
  end
  nAnomaly = size(xAnomalyRep,1);
    
  % selection of subspaces
  for m = 1:length(methods)
    fprintf('##### Running method: %s....\n', methods{m});
    performance{m}.method = methods{m};
    if strcmpi(methods{m}, 'aag')
      fprintf('##### Discretizing attributes ...\n');
      [xTrain_d, ~] = discretize_data(xTrainRep);
      fprintf('##### Performing subspace analysis with AAG ...\n');
      [T, ~] = maag4(xTrain_d, [], 0, 0, 0, 1, 1, 1);
      % clean up identical groups for anomaly detection ensemble
      G = cleanup_groups(T);

    elseif strcmpi(methods{m}, 'fb')
      G = fb_subspace(xTrainRep);
      
    elseif strcmpi(methods{m}, 'enclus')
      fprintf('##### Discretizing attributes ...\n');
      [xTrain_d, ~] = discretize_data(xTrainRep);
      f_enclus1 = 0.5;
      f_enclus2 = 1.5;
      fprintf('##### Performing subspace analysis with ENCLUS ...\n');
      G = enclus_subspaces(xTrain_d, f_enclus1, f_enclus2);
      
    elseif strcmpi(methods{m}, 'ewkm')
      fprintf('##### Performing subspace analysis with EWKM ...\n');
      G = ewkm_subspace(xTrainRep);
      
    elseif strcmpi(methods{m}, 'afg')
      fprintf('##### Performing subspace analysis with AFG-kMeans ...\n');
      G = afg_k_means_subspace(xTrainRep);
      
    elseif strcmpi(methods{m}, 'iforest')
      fprintf('##### Performing subspace analysis with iForest ...\n');
      NumTree = 100; % number of isolation trees
      NumSub = max(20, min(floor(nTrain/4),max(256, nTrain))); % subsample size
      NumDim = size(xTrainRep, 2); % do not perform dimension sampling
      iTree = IsolationForest(xTrainRep, NumTree, NumSub, NumDim, ...
        randi(10000));
      auc = zeros(1,n_noise_factor);
      f1 = zeros(1, n_noise_factor);
      
      for k=1:n_noise_factor
        if add_noise == 1
          dim = randsample(P, min(P, round(noise_factors(k)*P+1)));
          fprintf('##### Noise corruption setting on %d attributes...\n', dim);
          noise_std = std(xAnomaly_orig(:,dim));
          
          xAnomalyRep(:, dim) = bsxfun(@plus, xAnomaly_orig(:,dim), ...
            bsxfun(@times, noise_std, mvnrnd(zeros(length(dim),1), ...
            eye(length(dim)), n_noise)));
        end
        [Mass, ~] = IsolationEstimation([xTestRep; xAnomalyRep], iTree);
        Score = mean(Mass, 2);
        labels = boolean([ones(nTest,1); zeros(nAnomaly,1)]);
        auc(k) = scoreAUC(labels, Score);
        index_cut = max(1, floor(alpha*length(labels)));
        [score_sorted, ~] = sort(Score,'ascend');
        score_cut = score_sorted(index_cut);
        yhat = double(Score >= score_cut);
        tp = sum((yhat == 0) & (labels == 0));
        fp = sum((yhat == 0) & (labels == 1));
        fn = sum((yhat == 1) & (labels == 0));
% %         tn = sum((yhat == 1) & (labels == 1));
        f1(k) = 2 * tp / (2 * tp + fp + fn);
      end
    else
      error('ERROR!!! %s not implemented!!', methods{m});
    end
    
    % build anomaly detection models for ensemble
    if ~strcmpi(methods{m}, 'iforest')
      subspace_weights = [];
      yhat = [];
      mvset_model = cell(1,1);
      model_pca = cell(1,1);
      f1 = zeros(1, n_noise_factor);
      auc = zeros(1, n_noise_factor);
      w = [];
      for g = 1:length(G)
        fprintf('##### MV-set model on subspace %d / %d ...\n', g, length(G));
        fs = G{g};
        result = [];
        for ival = 1:nRepVal
          % validation dataset only for MV-set model
          if nTrain > 200
            selVal = zeros(nTrain,1);
            selVal(randsample(nTrain, round(.3*nTrain))) = 1;
            xTrainAn = xTrainRep(selVal == 0,:);
            xVal = xTrainRep(selVal>0,:);
            nVal = size(xVal,1);
            % record indices of selected data
            i_val = find(selVal > 0);
          else
            xTrainAn = xTrainRep;
            xVal = xTrainRep;
            nVal = nTrain;
            i_val = (1:nVal)';
          end
          [i_result, ~, ~, model_pca{g}, mvset_model{g}, ...
            ~, ~] = find_mv_set(xTrainAn(:,fs), ...
            [xVal(:,fs); xTestRep(:,fs); xAnomalyRep(:,fs)], 0.9, alpha);
          if ~isempty(mvset_model{g})
            result = [result i_result];
          end
        end
        subspace_weights = [subspace_weights ...
          sum((sum(result(1:nVal,:),2)/nRepVal))/nVal];
        % used for subspace weight factors
        result = double((sum(result,2) / nRepVal) > 0.5);
        % retrain using the whole traininig data
        [~, ~, ~, model_pca{g}, mvset_model{g}, ~, ~] = ...
          find_mv_set(xTrainRep(:,fs), xAnomalyRep(:,fs), 0.9, alpha);
        if isempty(mvset_model{g})
          warning('MV-set model could not be computed!');
        else
          % compute subspace weitghs
          if add_noise == 0
            yhat = [yhat result];
          end
        end
      end
      
      % if noise setting is used, then we just use the trained models and
      % apply them to the corrupted data. If other settings are used, then
      % we also apply the trained model to compute the performances nut
      % only once.
      for k=1:n_noise_factor
        if add_noise == 1
          dim = randsample(P, min(P, round(noise_factors(k)*P+1)));
          fprintf('##### Noise corruption setting on %d attributes...\n', dim);
          noise_std = std(xAnomaly_orig(:,dim));
          
          xAnomalyRep(:, dim) = bsxfun(@plus, xAnomaly_orig(:,dim), ...
            bsxfun(@times, noise_std, mvnrnd(zeros(length(dim),1), ...
            eye(length(dim)), n_noise)));
          
          yhat = [];
          for g=1:length(G)
            fs = G{g};
            if ~isempty(mvset_model{g})
              x = [xTrain(:,fs); xTestRep(:,fs); xAnomalyRep(:,fs)];
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
            subspace_weights, [ones(nTrain+nTest,1); ones(nAnomaly,1)], ...
            nTrain, nTest, nAnomaly);
          f1(k) = f1_noise;
          auc(k) = auc_noise;
          w = [w w_noise];
        else
          [f1, auc, w, ~, ~] = calc_performance(yhat, ...
            subspace_weights, [ones(nVal+nTest,1); ones(nAnomaly,1)], ...
            nVal, nTest, nAnomaly);
        end
      end
    end
    % record results    
    fprintf('##### Recording performance...\n');
    performance{m}.results(r).f1 = f1;
    performance{m}.results(r).auc = auc;
    performance{i}.results(r).train = i_train;
    performance{i}.results(r).val = i_val;
    performance{i}.results(r).test = i_test;
    performance{i}.results(r).anomaly = i_anomaly;
    if ~strcmpi(methods{m}, 'iforest')
      performance{m}.results(r).mvmodel = mvset_model;
      performance{m}.results(r).pca = model_pca;
      performance{m}.results(r).subspaces = G;
      performance{m}.results(r).subspaces_weights = subspace_weights;
      performance{m}.results(r).w = w;
    else
      performance{m}.results(r).mvmodel = iTree;      
    end
  end
end










