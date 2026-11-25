function specs = build_final_region_specs(bounds)
%% Build the final left, center, and right region specifications.

assert(isequal(size(bounds), [3 2]) && all(isfinite(bounds), 'all'), ...
    'Region bounds must be a finite 3-by-2 matrix.');
assert(all(bounds(:,1) < bounds(:,2)) && ...
    abs(bounds(1,2)-bounds(2,1)) <= 1e-14 && ...
    abs(bounds(2,2)-bounds(3,1)) <= 1e-14, ...
    'Region bounds must form a contiguous partition.');

names = {'left','center','right'};
labels = {'OTM Put','Near ATM','OTM Call'};
specs = repmat(struct('name','','label','','bounds',[]), 3, 1);
for j = 1:3
    specs(j).name = names{j};
    specs(j).label = labels{j};
    specs(j).bounds = bounds(j,:);
end

end
