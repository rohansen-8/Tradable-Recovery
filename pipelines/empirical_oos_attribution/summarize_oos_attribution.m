function S = summarize_oos_attribution(T, meta)
%% OOS performance and attribution statistics with HAC inference (saved outputs only).
%
% Return definitions (never mixed within one comparison):
%   as implemented   total = alpha' h (the strategy whose SR the paper reports);
%                    forward leg = h_m(1,v) * SP500_ret (index PRICE return,
%                    unfunded, no dividends), option legs financed at mid.
%   forward contract the same positions with the forward leg on the forward
%                    excess payoff SP500_ret - carry_hold = (S_next - F)/S,
%                    i.e. total - h_m * carry_hold. carry_hold is known at
%                    formation, so positions are unchanged.
%   market (as implemented) Delta = SP500_ret, one unit of the index under the
%                    same r = 0 normalization as the forward leg (primary
%                    benchmark);
%   market           forward excess  Delta - carry_hold (dividend-inclusive
%                    excess return under the implied forward), and, separately,
%                    price excess Delta - rf_hold (ex-dividend).
% Inference: Bartlett HAC with the lag rule bartlett_hac_bandwidth(T).
% Sharpe ratios are mean/std(ddof=1)*sqrt(N), N from the protocol
% (manuscript convention); delta-method standard errors are asymptotic.
% Pre-specified statistics: SRs and Delta SR; leg means and Euler
% shares; option leg on forward leg and on market; forward position vs VIX
% and timing share; near-ATM leg by initial-window VIX terciles; lowest-strike
% legs in the worst decile of market outcomes.

N = meta.protocol.annualization;
n = height(T);
L = bartlett_hac_bandwidth(n);
w = T.fwd_position;
carry = w .* T.carry_hold;
legs = local_legs(T);

r = struct();
r.total_impl = T.total;
r.total_fwdcontract = T.total - carry;
r.forward_impl = T.forward;
r.forward_excess = T.forward - carry;
r.options = T.total - T.forward;
r.market_impl = T.Delta;
r.market_fwd_excess = T.Delta - T.carry_hold;
r.market_price_excess = T.Delta - T.rf_hold;
names = fieldnames(r);

SR = table('Size', [numel(names) 4], 'VariableTypes', {'string','double','double','double'}, ...
    'VariableNames', {'series','mean_ann','SR_ann','SR_se'});
for j = 1:numel(names)
    x = r.(names{j});
    [s, se] = sharpe_hac(x, N, L);
    SR(j, :) = {names{j}, mean(x) * N, s, se};
end

pairs = {'total_impl','forward_impl'; 'total_impl','market_impl'; ...
    'forward_impl','market_impl'; 'total_fwdcontract','forward_excess'; ...
    'total_fwdcontract','market_fwd_excess'};
dSR = table('Size', [size(pairs,1) 3], 'VariableTypes', {'string','double','double'}, ...
    'VariableNames', {'comparison','dSR_ann','se'});
for j = 1:size(pairs, 1)
    [d, se] = dsharpe_hac(r.(pairs{j,1}), r.(pairs{j,2}), N, L);
    dSR(j, :) = {sprintf('%s - %s', pairs{j,1}, pairs{j,2}), d, se};
end

% Leg means and Euler shares of the as-implemented total.
legNames = fieldnames(legs);
P = r.total_impl;
Legs = table('Size', [numel(legNames) 5], 'VariableTypes', {'string','double','double','double','double'}, ...
    'VariableNames', {'leg','mean_ann','t_HAC','mean_share','var_share'});
for j = 1:numel(legNames)
    x = legs.(legNames{j});
    [b, se] = ols_hac(x, ones(n,1), L);
    Legs(j, :) = {legNames{j}, b * N, b / se, mean(x) / mean(P), cov_pop(x, P) / cov_pop(P, P)};
end

% Spanning regressions (alpha per period annualized, HAC t).
reg = @(y, x) ols_hac(y, [ones(n,1) x], L);
Reg = table('Size', [6 5], 'VariableTypes', {'string','double','double','double','double'}, ...
    'VariableNames', {'regression','alpha_ann','alpha_t','beta','beta_t'});
specs = {'total_impl on market_impl', r.total_impl, r.market_impl; ...
         'forward_impl on market_impl', r.forward_impl, r.market_impl; ...
         'options on market_impl', r.options, r.market_impl; ...
         'options on forward_impl', r.options, r.forward_impl; ...
         'options on market_fwd_excess', r.options, r.market_fwd_excess; ...
         'forward_excess on market_fwd_excess', r.forward_excess, r.market_fwd_excess};
for j = 1:size(specs, 1)
    [b, se] = reg(specs{j,2}, specs{j,3});
    Reg(j, :) = {specs{j,1}, b(1) * N, b(1) / se(1), b(2), b(2) / se(2)};
end

% (A) State dependence of the OOS forward position.
[bA, seA] = ols_hac(w, [ones(n,1) T.vix], L);
A = struct('corr_position_vix', corr_pop(w, T.vix), 'slope_on_vix', bA(2), 't_slope', bA(2) / seA(2), ...
    'timing_share', mean((w - mean(w)) .* T.Delta) / mean(w .* T.Delta));

% (B) Near-ATM bucket (b3) by initial-window VIX terciles.
q = local_quantile(meta.initial_window_vix, [1/3 2/3]);
g = 1 + (T.vix > q(1)) + (T.vix > q(2));
mid = legs.b3_c1_to_c2;
B = struct('vix_cut', q, 'mean_ann_by_tercile', accumarray(g, mid, [3 1], @mean).' * N, ...
    'n_by_tercile', accumarray(g, 1, [3 1]).', 'corr_mid_forward', corr_pop(mid, r.forward_impl), ...
    'var_ratio_without_mid', var(P - mid) / var(P), ...
    'SR_without_mid', mean(P - mid) / std(P - mid) * sqrt(N));
[bB, seB] = ols_hac(mid, [ones(n,1) r.market_fwd_excess r.market_fwd_excess.^2 - mean(r.market_fwd_excess.^2)], L);
B.intercept_ann = bB(1) * N; B.intercept_t = bB(1) / seB(1);
B.loading_sq = bB(3); B.loading_sq_t = bB(3) / seB(3);

% Lowest-strike legs (b1 + b2) in the worst decile of market outcomes.
low = legs.b1_below_domain + legs.b2_a_to_c1;
cut = local_quantile(r.market_fwd_excess, 0.10);
worst = r.market_fwd_excess <= cut;
C = struct('market_cut', cut, 'n_worst', sum(worst), ...
    'low_mean_ann_worst', mean(low(worst)) * N, 'low_mean_ann_other', mean(low(~worst)) * N, ...
    'skew_total', skew(P), 'skew_total_without_low', skew(P - low), ...
    'q05_total', local_quantile(P, 0.05), 'q05_total_without_low', local_quantile(P - low, 0.05));

S = struct('n', n, 'hac_lag', L, 'annualization', N, 'sharpe', SR, 'delta_sharpe', dSR, ...
    'legs', Legs, 'regressions', Reg, 'finding_A', A, 'finding_B', B, 'low_strikes_tail', C, ...
    'identity_max_abs', max(abs([T.identity1; T.identity2; T.identity3])), ...
    'bucket_labels', {meta.bucket_labels}, 'periods_per_year_realized', ...
    n / (days(T.oos_date(end) - T.oos_date(1)) / 365.25));

end

function legs = local_legs(T)
%% Forward leg, strike buckets (put+call), and the outside-domain total.
legs = struct();
legs.forward = T.forward;
legs.b1_below_domain = T.c1_put + T.c1_call;
legs.b2_a_to_c1 = T.c2_put + T.c2_call;
legs.b3_c1_to_c2 = T.c3_put + T.c3_call;
legs.b3_puts = T.c3_put;
legs.b3_calls = T.c3_call;
legs.b4_c2_to_z = T.c4_put + T.c4_call;
legs.b5_above_domain = T.c5_put + T.c5_call;
legs.outside_test_domain = legs.b1_below_domain + legs.b5_above_domain;
end

function [s, se] = sharpe_hac(x, N, L)
%% Annualized Sharpe ratio (ddof = 1) and its HAC delta-method s.e.
x = x(:); n = numel(x);
s = mean(x) / std(x) * sqrt(N);
mu = mean(x); m2 = mean(x.^2); sg = sqrt(m2 - mu^2);
g = [m2 / sg^3; -mu / (2 * sg^3)];
V = nw_cov([x, x.^2], L);
se = sqrt(g' * V * g / n) * sqrt(N);
end

function [d, se] = dsharpe_hac(x, y, N, L)
%% Difference of annualized Sharpe ratios of paired series, HAC delta method.
x = x(:); y = y(:); n = numel(x);
d = (mean(x) / std(x) - mean(y) / std(y)) * sqrt(N);
gx = grad_sr(x); gy = grad_sr(y);
g = [gx; -gy];
V = nw_cov([x, x.^2, y, y.^2], L);
se = sqrt(g' * V * g / n) * sqrt(N);
end

function g = grad_sr(x)
%% Gradient of mu/sqrt(m2-mu^2) with respect to (mu, m2).
mu = mean(x); m2 = mean(x.^2); sg = sqrt(m2 - mu^2);
g = [m2 / sg^3; -mu / (2 * sg^3)];
end

function [b, se] = ols_hac(y, X, L)
%% OLS with Bartlett-HAC standard errors.
n = numel(y);
b = X \ y;
e = y - X * b;
Sxx = (X' * X) / n;
V = (Sxx \ nw_cov(X .* e, L, false)) / Sxx / n;
se = sqrt(diag(V));
end

function V = nw_cov(G, L, demean)
%% Newey-West (Bartlett) long-run covariance of the rows of G.
if nargin < 3 || demean, G = G - mean(G, 1); end
n = size(G, 1);
V = (G' * G) / n;
for l = 1:L
    C = (G(l+1:end, :)' * G(1:end-l, :)) / n;
    V = V + (1 - l / (L + 1)) * (C + C');
end
end

function c = cov_pop(x, y)
%% Population covariance.
c = mean((x - mean(x)) .* (y - mean(y)));
end

function c = corr_pop(x, y)
%% Pearson correlation.
c = cov_pop(x, y) / sqrt(cov_pop(x, x) * cov_pop(y, y));
end

function s = skew(x)
%% Sample skewness (population moments).
z = (x - mean(x)) / sqrt(cov_pop(x, x));
s = mean(z.^3);
end

function q = local_quantile(x, p)
%% Empirical quantile, order statistic x_(ceil(p*n)) (no toolbox dependency).
v = sort(x(:));
q = v(max(1, ceil(p * numel(v))));
q = reshape(q, size(p));
end
