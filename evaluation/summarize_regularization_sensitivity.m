function summary = summarize_regularization_sensitivity(raw)
%% Summarize fixed-dataset regularization sensitivity across replications.
%
% The primary CV-fit HAC metric is held fixed within each dataset; lambda is
% varied only in the fitted shape vector.

if isempty(raw)
    summary = table();
    return
end

hasDesign = ismember('design_id', raw.Properties.VariableNames);
if hasDesign
    [G, design_id, design_group, Tval, lambda, region, testType, role, distance_metric] = findgroups( ...
        raw.design_id, raw.design_group, raw.Tval, raw.lambda, raw.region, ...
        raw.testType, raw.role, raw.distance_metric);
else
    [G, Tval, lambda, region, testType, role, distance_metric] = findgroups( ...
        raw.Tval, raw.lambda, raw.region, raw.testType, raw.role, raw.distance_metric);
end

B = splitapply(@(x) numel(unique(x)), raw.seed, G);
lambda_cv_median = splitapply(@median, raw.lambda_cv, G);
min_theta_median = splitapply(@median, raw.min_theta, G);
min_theta_p25 = splitapply(@(x) quantile(x, 0.25), raw.min_theta, G);
min_theta_p75 = splitapply(@(x) quantile(x, 0.75), raw.min_theta, G);
frac_negative_mean = splitapply(@mean, raw.frac_negative, G);
euclidean_median = splitapply(@median, raw.euclidean_cone_distance, G);
euclidean_p25 = splitapply(@(x) quantile(x, 0.25), raw.euclidean_cone_distance, G);
euclidean_p75 = splitapply(@(x) quantile(x, 0.75), raw.euclidean_cone_distance, G);
mahalanobis_median = splitapply(@median, raw.mahalanobis_cone_distance, G);
mahalanobis_p25 = splitapply(@(x) quantile(x, 0.25), raw.mahalanobis_cone_distance, G);
mahalanobis_p75 = splitapply(@(x) quantile(x, 0.75), raw.mahalanobis_cone_distance, G);

if hasDesign
    summary = table(design_id, design_group, Tval, B, lambda, lambda_cv_median, ...
        region, testType, role, distance_metric, min_theta_median, ...
        min_theta_p25, min_theta_p75, frac_negative_mean, euclidean_median, ...
        euclidean_p25, euclidean_p75, mahalanobis_median, ...
        mahalanobis_p25, mahalanobis_p75);
    summary = sortrows(summary, {'design_id','testType','region','lambda'}, ...
        {'ascend','ascend','ascend','descend'});
else
    summary = table(Tval, B, lambda, lambda_cv_median, region, testType, role, ...
        distance_metric, min_theta_median, min_theta_p25, min_theta_p75, ...
        frac_negative_mean, euclidean_median, euclidean_p25, euclidean_p75, ...
        mahalanobis_median, mahalanobis_p25, mahalanobis_p75);
    summary = sortrows(summary, {'Tval','testType','region','lambda'}, ...
        {'ascend','ascend','ascend','descend'});
end

end
