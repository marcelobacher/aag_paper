function G_clean = cleanup_groups(G)
l = 1;
G_clean = cell(1);
for k=1:length(G)
  a = G{k};
  do_append = ones(length(a),1);
  if ~isempty(G_clean{1})
    % check if the group is already there
    for ii=1:length(a)
      if ~isempty(a{ii})
        for u=1:length(G_clean)
          if jaccard_index(G_clean{u}, a{ii}) == 1
            % do not append
            do_append(ii) = 0;
            break;
          end
        end
      end
    end
  end
  
  for i=1:length(a)
    if ~isempty(a{i}) && (do_append(i) == 1)
      G_clean{l} = a{i};
      l = l + 1;
    end
  end
end