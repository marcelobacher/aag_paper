%%
% S = enclus(data, w, e, Smax, gain, verbose)
%
% This function implements ENCLUS (Entropy Subspace Clusterin) described in
% the paper:
%
% Chun-Hung Cheng, Ada Waichee Fu, and Yi Zhang, Entropy-based subspace 
% clustering for mining numerical data. In Proceedings of the fifth ACM 
% SIGKDD international conference on Knowledge discovery and
% data mining, pages 84?93. ACM Press, 1999.
%
% Input:
%   data: Data matrix of dimension n x d, where n is the number of
%   instances and d the data dimensionality (obligatory)
%   w:    Entropy threshold to select potential subspaces (obligatory)
%   e:    Threshold to select interesting subspaces (obligatory)
%   Smax: Maximal subspace dimensionality (optional, default 6)
%   gain: (=0) Significant subspace, (=1) interesting subspaces
%
% Output:
%   S:    cell array with selected subspaces
%
% Marcelo Bacher, June 2016
%===============================
function S = enclus(data, w, e1, e2, Smax, gain, verbose)

% set of subspaces to return
S = cell(1,1);

% data parameters
d = size(data, 2);

% flag to control loop
keep_looking = 1;

% single entropy values to save run-time
H0 = zeros(d,1);
for c=1:d
  H0(c) = mvJointEntropy(data(:,c));
end

% set of initial candidates
C = (1:d)';

% solution counter
k = 1;

% loop counter
loop = 1;

if verbose
  disp('ENCLUS: starting main loop...');
end

% main loop
while keep_looking
  % number of candidates in loop k
  nC = size(C,1);
  H = zeros(nC,1);
  % solution subspaces in loop k
  Stmp = [];
  % subspaces to be combined in loop k+1
  NStmp = [];
  
  if verbose
    disp(['ENCLUS: Loop: ' num2str(loop) ' \\ Candidates: ' num2str(nC) ...
      ' \\ Dimension: ' num2str(size(C,2))]);
  end
  for c = 1:nC
    if length(C(c,:)) < Smax
      H(c) = sum(abs(mvJointEntropy(data(:,C(c,:)),0)));
      if (H(c) < w) && ((length(C(c,:))==1) || ...
          (interest(H(c), H0, C(c,:), gain) >= ...
          e1*(Smax/(1+Smax-length(C(c,:))))))
        if interest(H(c), H0, C(c,:), gain) > e2
          Stmp = [Stmp; C(c,:)];
        else
          NStmp = [NStmp; C(c,:)];
        end
      end
    end
  end

  % add subspaces to the list
  if ~isempty(Stmp)
    S{k} = Stmp;
    k = k + 1;
  end
  
  % generate list of candidate subspaces
  if gain == 0
    %  significant subspaces
    C = generate_canditates(NStmp);
  else
    % interesting subspaces
    C = generate_canditates([Stmp; NStmp]);
  end
  
  if isempty(C)
    % finish algorithm
    keep_looking = 0;
  else
    loop = loop + 1;
  end
end

% compute total correlation aka interest
function tc = interest(h, H0, index, gain)
tc = sum(H0(index)) - h;
if gain > 0 && (length(index)>1)
  maxi = -Inf;
  for i=1:length(index)
    sel = ones(length(index),1);
    sel(i) = 0;
    a = sum(H0(index(sel>0))) - h;
    if a > maxi
      maxi = a;
    end
  end
  tc = tc - a;
end


% generate a list of candidates based on NS set up to dimensionality Smax
function C = generate_canditates(NS)
C = [];
if ~isempty(NS) && size(NS,1)>1
  for p=1:size(NS,1)-1
    try
    for q=min(p+1, size(NS,1)):size(NS,1)
      if length(NS(p,:)) == 1
        C = [C; union(NS(p,:), NS(q,:))];
      else
        check4add = 0;
        
        for i=1:length(NS(p,:))-1
          if NS(p,i) == NS(q,i)
            check4add = check4add + 1;
          end
        end
        if (check4add == length(NS(p,:))-1) && ...
           (NS(p,i+1) < NS(q,i+1))
          C = [C; union(NS(p,:), NS(q,:))];
        end
      end
    end
            catch e
          disp (e);
        end
  end
  C = unique(C, 'rows');
end