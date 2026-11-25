function test_summarize_bs_rejection_results()
%% Verify rejection aggregation and lambda-boundary diagnostics.

setpath();

n = 4;
horizon = repmat("monthly", n, 1);
Tval = repmat(273, n, 1);
region = repmat("center", n, 1);
testType = repmat("conv", n, 1);
lambda_mode = repmat("cv", n, 1);
covariance_rule = repmat("hac_bartlett_L4", n, 1);
covariance_type = repmat("hac_bartlett", n, 1);
covariance_bandwidth = repmat(4, n, 1);
covariance_bandwidth_rule = repmat("", n, 1);
winsor_margin = zeros(n, 1);

lambda = [1e-5; 1e-4; 1e-2; 1e-2];
p_value = [0.02; 0.10; 0.01; 0.20];
reject_1pct = [0; 0; 0; 0];
reject_5pct = [1; 0; 1; 0];
reject_10pct = [1; 0; 1; 0];
cv_mean_loss = [1.0; 0.9; 0.8; 0.7];

raw = table(horizon, Tval, region, testType, lambda_mode, ...
    covariance_rule, covariance_type, covariance_bandwidth, ...
    covariance_bandwidth_rule, winsor_margin, lambda, p_value, ...
    reject_1pct, reject_5pct, reject_10pct, cv_mean_loss);

summary = summarize_bs_rejection_results(raw, logspace(-5, -2, 31).');

assert(height(summary) == 1);
assert(summary.B == 4);
assert(summary.valid_tests == 4);
assert(abs(summary.lambda_lower_bound_freq - 0.25) < 1e-14);
assert(abs(summary.lambda_upper_bound_freq - 0.50) < 1e-14);
assert(abs(summary.rej_5pct - 0.50) < 1e-14);
assert(abs(summary.cv_mean_loss - 0.85) < 1e-14);

fprintf('test_summarize_bs_rejection_results passed.\n');

end
