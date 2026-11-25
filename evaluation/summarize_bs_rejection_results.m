function summary = summarize_bs_rejection_results(raw, lambda_grid)
%% Aggregate BS replication results into rejection and lambda diagnostics.

if nargin < 2 || isempty(lambda_grid)
    error('summarize_bs_rejection_results:MissingGrid', ...
        'Provide the lambda grid used in the simulation design.');
end

lambda_grid = lambda_grid(:);
lambdaLower = min(lambda_grid);
lambdaUpper = max(lambda_grid);

lambdaCoordinate = raw.lambda;
if ismember('lambda_multiplier', raw.Properties.VariableNames)
    lambdaMultiplier = raw.lambda_multiplier;
else
    lambdaMultiplier = raw.lambda;
end
if ismember('lambda_fit', raw.Properties.VariableNames)
    lambdaFit = raw.lambda_fit;
else
    lambdaFit = raw.lambda;
end

[G, horizon, Tval, region, testType, lambda_mode, covariance_rule, covariance_type, ...
    covariance_bandwidth, covariance_bandwidth_rule, winsor_margin] = findgroups( ...
    raw.horizon, raw.Tval, raw.region, raw.testType, raw.lambda_mode, ...
    raw.covariance_rule, raw.covariance_type, raw.covariance_bandwidth, ...
    raw.covariance_bandwidth_rule, raw.winsor_margin);

B = splitapply(@numel, raw.p_value, G);
valid_tests = splitapply(@(x) sum(isfinite(x)), raw.p_value, G);
rej_1pct = splitapply(@(x) mean(x, 'omitnan'), raw.reject_1pct, G);
rej_5pct = splitapply(@(x) mean(x, 'omitnan'), raw.reject_5pct, G);
rej_10pct = splitapply(@(x) mean(x, 'omitnan'), raw.reject_10pct, G);

se_1pct = sqrt(rej_1pct .* (1 - rej_1pct) ./ valid_tests);
se_5pct = sqrt(rej_5pct .* (1 - rej_5pct) ./ valid_tests);
se_10pct = sqrt(rej_10pct .* (1 - rej_10pct) ./ valid_tests);

lambda_median = splitapply(@median, raw.lambda, G);
lambda_inference_median = lambda_median;
lambda_p25 = splitapply(@(x) quantile(x, 0.25), raw.lambda, G);
lambda_p75 = splitapply(@(x) quantile(x, 0.75), raw.lambda, G);
lambda_multiplier_median = splitapply(@median, lambdaMultiplier, G);
lambda_multiplier_p25 = splitapply(@(x) quantile(x, 0.25), lambdaMultiplier, G);
lambda_multiplier_p75 = splitapply(@(x) quantile(x, 0.75), lambdaMultiplier, G);
lambda_lower_bound_freq = splitapply(@(x) mean(abs(x - lambdaLower) <= 1e-12 * max(1, abs(lambdaLower))), lambdaCoordinate, G);
lambda_upper_bound_freq = splitapply(@(x) mean(abs(x - lambdaUpper) <= 1e-12 * max(1, abs(lambdaUpper))), lambdaCoordinate, G);
lambda_fit_median = splitapply(@median, lambdaFit, G);
lambda_fit_lower_bound_freq = splitapply(@(x) mean(abs(x - lambdaLower) <= 1e-12 * max(1, abs(lambdaLower))), lambdaFit, G);
lambda_fit_upper_bound_freq = splitapply(@(x) mean(abs(x - lambdaUpper) <= 1e-12 * max(1, abs(lambdaUpper))), lambdaFit, G);
lambda_changed_freq = splitapply(@(x) mean(x), double(lambdaCoordinate < lambdaFit), G);
lambda_ratio_median = splitapply(@median, lambdaCoordinate ./ lambdaFit, G);
cv_mean_loss = splitapply(@(x) mean(x, 'omitnan'), raw.cv_mean_loss, G);

summary = table(horizon, Tval, B, valid_tests, region, testType, lambda_mode, ...
    lambda_median, lambda_p25, lambda_p75, lambda_multiplier_median, ...
    lambda_multiplier_p25, lambda_multiplier_p75, lambda_lower_bound_freq, ...
    lambda_upper_bound_freq, lambda_inference_median, lambda_fit_median, lambda_fit_lower_bound_freq, ...
    lambda_fit_upper_bound_freq, lambda_changed_freq, lambda_ratio_median, ...
    cv_mean_loss, covariance_rule, covariance_type, ...
    covariance_bandwidth, covariance_bandwidth_rule, winsor_margin, ...
    rej_1pct, se_1pct, rej_5pct, se_5pct, rej_10pct, se_10pct, ...
    'VariableNames', {'horizon', 'Tval', 'B', 'valid_tests', 'region', 'testType', ...
    'lambda_mode', 'lambda_median', 'lambda_p25', 'lambda_p75', ...
    'lambda_multiplier_median', 'lambda_multiplier_p25', 'lambda_multiplier_p75', ...
    'lambda_lower_bound_freq', 'lambda_upper_bound_freq', ...
    'lambda_inference_median', 'lambda_fit_median', 'lambda_fit_lower_bound_freq', ...
    'lambda_fit_upper_bound_freq', 'lambda_changed_freq', 'lambda_ratio_median', ...
    'cv_mean_loss', ...
    'covariance_rule', 'covariance_type', 'covariance_bandwidth', ...
    'covariance_bandwidth_rule', 'winsor_margin', ...
    'rej_1pct', 'se_1pct', 'rej_5pct', 'se_5pct', 'rej_10pct', 'se_10pct'});

summary = sortrows(summary, {'horizon', 'Tval', 'testType', 'region'});

end
