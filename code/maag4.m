function [T, Tg] = maag4(X, y, w1, w2, psi, c, pruning, verbose)
  % create dictionary to be filled in with distances, where keys are
  % attribute combinations
  distances = containers.Map();
  T = cell(1,1);
  p = size(X,2);
  Tg = [];
  % track remaining subspaces
  S_current = cell(p,1);
  % current agglomeration level
  t = 1;
  % intialization of level 0
  for i1 = 1:p
    S_current{i1} = i1;
  end
  % pre-compute entropies to be used later to check for pruning conditions
  H = size(p,1);
  for i=1:p
    H(i) = mvJointEntropy(X(:,i));
  end
  
  if verbose
    disp('... initializing algorithm...');
  end
  while get_num_subspaces(S_current) > 1
    if verbose
      disp(['... Building level ' num2str(t) '...']);
    end
    % save old subspaces
    S0 = S_current;
    S_next = [];
    % choose first two subspaces at level t
    [comb_subspaces, comb_subspaces_map] = ...
      generate_combination(S_current, []);
    % find subspace combination with minimal distance
    min_d = Inf;
    min_d_i = 0;
    for i1 = 1:size(comb_subspaces,1)
%       si = get_relevant_attributes(comb_subspaces{i1,1});
%       sj = get_relevant_attributes(comb_subspaces{i1,2});
%       [d, distances] = calc_distance(X, y, w1, w2, psi, c, ...
%         unique([si sj]), distances);
        [d, distances] = calc_distance(X, y, w1, w2, psi, c, ...
        unique([comb_subspaces{i1,1} comb_subspaces{i1,2}]), distances);
      if d < min_d
        min_d = d;
        min_d_i = i1;
      end
    end
    % add subspace to next level (or check pruning)
    if t > 2 && pruning > 0
      g1 = comb_subspaces{min_d_i,1};
      g2 = comb_subspaces{min_d_i,2}; 
      if checkPruning(g1, g2, H)
        S_next = [S_next; {unique([comb_subspaces{min_d_i,1} ...
          comb_subspaces{min_d_i,2}])}];
      end
    else
      S_next = [S_next; {unique([comb_subspaces{min_d_i,1} ...
        comb_subspaces{min_d_i,2}])}];
    end
%     si = get_relevant_attributes(comb_subspaces{min_d_i,1});
%     sj = get_relevant_attributes(comb_subspaces{min_d_i,2});
%     S_next = [S_next; {unique([si sj])}];

    % remove combined subspaces from current level
    S_current(comb_subspaces_map(min_d_i,1)) = {[]};
    S_current(comb_subspaces_map(min_d_i,2)) = {[]};
    
    % next level empty, take next round
    if isempty(S_next)
      continue;
    end
    
    while get_num_subspaces(S_current) > 0
      if verbose>0
        disp(S_current)
        disp(S_next)
      end
      min_d = Inf;
      min_si = 0;
      min_sj = 0;
      min_si_idx = 0;
      min_sj_idx = 0;
      for i1=1:size(S_current,1)
%         si = get_relevant_attributes(S_current{i1});
        si = S_current{i1};
        if ~isempty(si)
          for j1=1:length(S_next)
%             sj = get_relevant_attributes(S_next{j1});
              sj = S_next{j1};
              [d, distances] = calc_distance(X, y, w1, w2, psi, c, ...
              unique([si sj]), distances);
            if d < min_d
              min_d = d;
              min_si = si;
              min_si_idx = i1;
              min_sj = sj;
              min_sj_idx = j1;
            end
          end
        end
      end
      min_d_k = Inf;
      min_sk = 0;
      for k1=1:size(S0,1)
        sk = S0{k1};
        if ~isempty(sk) && jaccard_index(sk, min_si) < 1
          [d_k, distances] = calc_distance(X, y, w1, w2, psi, c, ...
            unique([sk min_si]), distances);
          if d_k < min_d_k
            min_d_k = d_k;
            min_sk = sk;
          end
        end
      end
      if min_d_k <= min_d
        % group A_k and A_i
        if t > 2 && pruning > 0
          if checkPruning(min_sk, min_si, H)
            S_next = [S_next; {unique([min_sk min_si])}];
          end
        else
          S_next = [S_next; {unique([min_sk min_si])}];
        end
        % remove A_k from S_current (if it exists)
        for k=1:size(S_current,1)
          if ~isempty(S_current{k})
            if jaccard_index(min_sk, S_current{k}) == 1
              S_current(k) = {[]};
              break;
            end
          end
        end
      else
        % group A_j and A_i
        if t > 2 && pruning > 0
          if checkPruning(min_sj, min_si, H)
            S_next{min_sj_idx} = unique([min_sj min_si]);
          end
        else
          S_next{min_sj_idx} = unique([min_sj min_si]);
        end
      end
      S_current(min_si_idx) = {[]};
    end
    T{t} = S_next;
    S_current = S_next;
    t = t + 1;
  end
end 


function [comb_s, comb_s_map] = generate_combination(S, S_minus)
  if size(S,1) > 1
    comb_i = nchoosek(1:size(S,1), 2);
  else
    comb_i = S{1};
  end
  comb_s = [];
  comb_s_map = [];
  for j4=1:size(comb_i,1)
    if ~isempty(S{comb_i(j4,1)}) && ~isempty(S{comb_i(j4,2)})
      if ~isempty(S_minus)
        if jaccard_index(S{comb_i(j4,1)}, S_minus) < 1 && ...
             jaccard_index(S{comb_i(j4,2)}, S_minus) < 1
          comb_s = [comb_s; {S{comb_i(j4,1)} S{comb_i(j4,2)}}];
          comb_s_map = [comb_s_map; [comb_i(j4,1), comb_i(j4,2)]];
        end
      else
        comb_s = [comb_s; {S{comb_i(j4,1)} S{comb_i(j4,2)}}];
        comb_s_map = [comb_s_map; [comb_i(j4,1), comb_i(j4,2)]];
      end
    end
  end
end

function num = get_num_subspaces(S)
  n = length(S);
  num = 0;
  for m=1:n
    if ~isempty(S{m})
      num = num + 1;
    end
  end
end

function [d, distances] = calc_distance(X, y, w1, w2, psi, c, ...
  indices, distances)
  % d = -1;
  if isempty(y)
    n = length(indices);
    if n == 2
      comb = perms(indices);
      key = comb2key(comb);
      d = check_distance(distances, {key{1}});
      if d < 0
        d = rokhlin3([X(:,comb(1,1)) X(:,comb(1,2))]);
      end  
      distances(key{1}) = d;
      distances(key{2}) = d;
    else
      comb = nchoosek(indices, 3);
      key = comb2key(comb);
      d = check_distance(distances, key);
      for i2=1:size(comb,1)
        if d(i2) < 0
          d(i2) = rokhlin3(X(:,comb(i2,:)));
        end
        pcomb = perms(comb(i2,:));
        pkey = comb2key(pcomb);
        for l=1:size(pcomb,1)
          distances(pkey{l}) = d(i2);
        end
      end
      dm = 0;
      for k=1:length(unique(comb(:,1)))
        sel = comb(:,1) == indices(k);
        dm = dm + min(d(sel));
      end
      d = dm;
    end
  else
    n = length(indices);
    if n == 2
      comb = perms(indices);
      key = comb2key(comb);
      key_c = key;
      for k = 1:size(comb,1)
        key_c{k} = strcat([key_c{k} ',y='], num2str(c));
      end
      d1 = check_distance(distances, {key{1}});
      d2 = check_distance(distances, {key_c{1}});
      if d2 < 0
        d2 = rokhlin3([X(y==c,comb(1,1)) X(y==c,comb(1,2))]);
      end  
      if d1 < 0
        d1 = rokhlin3([X(:,comb(1,1)) X(:,comb(1,2))]);
      end
      distances(key{1}) = d1;
      distances(key{2}) = d1;
      distances(key_c{1}) = d2;
      distances(key_c{2}) = d2;

      key_fy = key;
      for k = 1:size(comb,1)
        key_fy{k} = strcat([key_fy{k} ',fy='], num2str(c));
      end
      d3 = check_distance(distances, key_fy);
      for k = 1:length(d3(:))
        if d3(k) < 0
          d3(k) = rokhlin3([X(:,comb(k,1)) X(:,comb(k,2)) y]);
          distances(key_fy{k}) = d3(k);
        end
      end
%       d3 = zeros(length(indices),1);
%       for k=1:length(indices)
%         key = [num2str(indices(k)) ',fy=', num2str(c)];
%         d3(k) = check_distance(distances, {key});
%         if d3(k) < 0
%           d3(k) = rokhlin3([X(:,indices(k)) y]);
%           distances(key) = d3(k);
%         end
%       end
      d = psi*d2 + (1-psi)*(w1*d1+w2*min(d3));
      %d = psi*d2 + (1-psi)*(w1*d1+w2*sum(d3));
    else
      comb = nchoosek(indices, 3);
      key = comb2key(comb);
      key_c = key;
      for k=1:size(comb,1)
        key_c{k} = [key{k} ',y=' num2str(c)];
      end      
      d1 = check_distance(distances, key_c);
      d2 = check_distance(distances, key);
      for i2=1:size(comb,1)      
        if d1(i2) < 0
          d1(i2) = rokhlin3(X(y==c,comb(1,:)));
          distances(key_c{i2}) = d1(i2);
        end
        
        if d2(i2) < 0
          d2(i2) = rokhlin3(X(:,comb(i2,:)));
          distances(key{i2}) = d2(i2);
        end
        pcomb = perms(comb(i2,:));
        pkey = comb2key(pcomb);
        for l=1:size(pcomb,1)
          distances(pkey{l}) = d2(i2);
          distances([pkey{l} ',y=' num2str(c)]) = d1(i2);
        end
      end      
      
      comb_c = nchoosek(indices, 2);
      key_c = comb2key(comb_c);
      for k=1:size(comb_c,1)
        key_c{k} = [key_c{k} ',fy=' num2str(c)];
      end
      d3 = check_distance(distances, key_c);
      for i2=1:size(comb_c,1)
        if d3(i2) < 0
          d3(i2) = rokhlin3([X(:,comb_c(i2,:)) y]);
          distances(key_c{i2}) = d3(i2);
        end
        pcomb_c = perms(comb_c(i2,:));
        pkey_c = comb2key(pcomb_c);
        for l=1:size(pcomb_c,1)
          distances([pkey_c{l} ',fy=' num2str(c)]) = d3(i2);
        end
      end
      
      d1min = 0;
      d2max = 0;
      for k=1:length(unique(comb(:,1)))
        sel = comb(:,1) == indices(k);
        d1min = d1min + min(d1(sel));
        d2max = d2max + max(d2(sel));
      end
      
      d = psi*d1min + (1-psi)*(w1*d2max+w2*min(d3));
      %d = psi*sum(d1) + (1-psi)*(w1*max(d2)+w2*min(d3));
      %d = psi*sum(d1) + (1-psi)*(w1*sum(d2)+w2*sum(d3));
    end    
  end
end


function val = check_distance(distances, key)
  val = [];
  for k = 1:length(key)
    if distances.isKey(key{k})
      val = [val; distances(key{k})];
    else
      val = [val; -1];
    end
  end
end


function keys = comb2key(comb)
  keys = [];
  for i3 = 1:size(comb,1)
    k = '';
    for j3=1:size(comb(i3,:),2)-1
        if isempty(k)
          k = [num2str(comb(i3,j3)) ','];
        else
          k = strcat(k,[num2str(comb(i3,j3)) ',']);
        end
    end
    k = strcat(k, num2str(comb(i3,j3+1)));
    keys = [keys; {k}];
  end
end


function do_group = checkPruning(g1, g2, H)
  g = union(g1,g2);
  h1 = (sum(H(g))-max(H(g)))/2;
  h2 = sum(H(g1)) - max(H(g1));
  h3 = sum(H(g2)) - max(H(g2));
  w1 = jaccard_index(g1, g);
  w2 = jaccard_index(g2, g);
  h = w1*h2 + w2*h3/ (w1+w2);
  do_group = double(h < h1);
end

function r = calc_dispersion_term(X)
  q = bsxfun(@minus, X, mean(X));
  r = 1/(eps + det(cov(q)));
end


function s = get_relevant_attributes(S)
  if length(S) > 3
    s = S(end:-1:end-1);
  elseif length(S) == 3
    s = S(2:3);
  else
    s = S;
  end
end