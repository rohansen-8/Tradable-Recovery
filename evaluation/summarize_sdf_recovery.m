function summary = summarize_sdf_recovery(raw)
%% Summarize pointwise production SDF-recovery differences.

if isempty(raw)
    summary = table();
    return
end

hasDesign = ismember('design_id', raw.Properties.VariableNames);
if hasDesign
    [G, design_id, design_group, Tval, m] = findgroups( ...
        raw.design_id, raw.design_group, raw.Tval, raw.m);
else
    [G, Tval, m] = findgroups(raw.Tval, raw.m);
end

B = splitapply(@(x) numel(unique(x)), raw.seed, G);
mean_difference = splitapply(@(x) mean(x, 'omitnan'), raw.difference, G);
median_difference = splitapply(@(x) median(x, 'omitnan'), raw.difference, G);
p05_difference = splitapply(@(x) quantile(x,0.05), raw.difference, G);
p25_difference = splitapply(@(x) quantile(x,0.25), raw.difference, G);
p75_difference = splitapply(@(x) quantile(x,0.75), raw.difference, G);
p95_difference = splitapply(@(x) quantile(x,0.95), raw.difference, G);
mean_estimated_sdf = splitapply(@(x) mean(x, 'omitnan'), raw.estimated_sdf, G);
mean_true_sdf = splitapply(@(x) mean(x, 'omitnan'), raw.true_sdf, G);

if hasDesign
    summary = table(design_id, design_group, Tval, B, m, mean_difference, ...
        median_difference, p05_difference, p25_difference, p75_difference, ...
        p95_difference, mean_estimated_sdf, mean_true_sdf);
    summary = sortrows(summary, {'design_id','m'});
else
    summary = table(Tval, B, m, mean_difference, median_difference, ...
        p05_difference, p25_difference, p75_difference, p95_difference, ...
        mean_estimated_sdf, mean_true_sdf);
    summary = sortrows(summary, {'Tval','m'});
end

end
