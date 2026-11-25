function res = lw_sharpe_difference_test(r, annualization, opts)
%% Ledoit-Wolf (2008) studentized time-series bootstrap test of SR1 - SR2.
%
%   res = lw_sharpe_difference_test(r, annualization, opts)
%
% r: T-by-2 paired returns (strategy, benchmark). opts fields (defaults):
%   M = 4999, K = 5000, Minner = 499, alpha = 0.05,
%   block_grid = [1 2 4 6 8 10] (restricted to b <= T/10),
%   seed_calibration = 2026081501, seed_final = 2026081502.
% p-values:
%   two-sided: (1 + #{|t*| >= |d|}) / (n_valid + 1)
%   one-sided H0: dSR <= 0: (1 + #{t* >= d}) / (n_valid + 1),
%   t* = (delta* - delta) / se*, d = delta / se (HAC).

if nargin < 3, opts = struct(); end
def = struct('M', 4999, 'K', 5000, 'Minner', 499, 'alpha', 0.05, ...
    'block_grid', [1 2 4 6 8 10], 'seed_calibration', 2026081501, ...
    'seed_final', 2026081502);
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}), opts.(fn{i}) = def.(fn{i}); end
end
T = size(r, 1);
blocks = opts.block_grid(opts.block_grid <= T / 10);
assert(~isempty(blocks), 'lw_sharpe_difference_test:NoBlocks', 'T too short for the block grid.');

h = lw_sharpe_hac_se(r);
cal = lw_sharpe_block_calibration(r, blocks, opts.K, opts.Minner, opts.alpha, ...
    RandStream('twister', 'Seed', opts.seed_calibration));
[delta_star, se_star] = lw_sharpe_boot_draws(r, cal.block_size, opts.M, ...
    RandStream('twister', 'Seed', opts.seed_final));
valid = isfinite(delta_star) & isfinite(se_star) & se_star > 0;
t_signed = (delta_star - h.delta) ./ se_star;
nv = sum(valid);
p_two = (1 + sum(abs(t_signed(valid)) >= abs(h.d))) / (nv + 1);
p_one = (1 + sum(t_signed(valid) >= h.d)) / (nv + 1);

res = struct('T', T, 'annualization', annualization, ...
    'sr_strategy_annual', h.sr(1) * sqrt(annualization), ...
    'sr_benchmark_annual', h.sr(2) * sqrt(annualization), ...
    'delta_sr_annual', h.delta * sqrt(annualization), ...
    'delta_sr_period', h.delta, 'se_period', h.se, 'd_observed', h.d, ...
    'block_size', cal.block_size, 'calibration', cal, ...
    'delta_star', delta_star, 'se_star', se_star, 'valid', valid, ...
    'n_valid', nv, 'p_value_one_sided', p_one, 'p_value_two_sided', p_two, ...
    'opts', opts);
end

function cal = lw_sharpe_block_calibration(r, blocks, K, Minner, alpha, stream, pseudoMeanBlock)
%% Ledoit-Wolf (2008) Algorithm 3.1: data-driven choice of the block size.
%
%   cal = lw_sharpe_block_calibration(r, blocks, K, Minner, alpha, stream)
%   cal = lw_sharpe_block_calibration(..., pseudoMeanBlock)   % default 10
%
% 1. Generate K pseudo sequences of length T by the stationary bootstrap
%    (Politis-Romano) of the observed paired returns, with geometric block
%    lengths of mean pseudoMeanBlock and circular wrapping.
% 2. For every pseudo sequence and every candidate block size b, form the
%    symmetric studentized bootstrap interval at level 1 - alpha
%    (Minner circular-block draws, lw_sharpe_boot_draws) and record
%    whether it covers the Sharpe difference of the observed sample, the
%    target value of the pseudo-data model.
% 3. Choose the block size whose coverage is closest to 1 - alpha; ties go
%    to the smaller block.
% The pseudo-data model is the stationary bootstrap with mean block length
% pseudoMeanBlock (default 10), and each interval uses Minner = 499 draws.

if nargin < 7 || isempty(pseudoMeanBlock)
    pseudoMeanBlock = 10;
end
T = size(r, 1);
target = lw_sharpe_hac_se(r).delta;
pNew = 1 / pseudoMeanBlock;
nb = numel(blocks);
covered = zeros(1, nb);
evaluated = zeros(1, nb);
for k = 1:K
    idx = zeros(T, 1);
    idx(1) = randi(stream, T);
    u = rand(stream, T, 1);
    fresh = randi(stream, T, T, 1);
    for t = 2:T
        if u(t) < pNew
            idx(t) = fresh(t);
        else
            idx(t) = mod(idx(t - 1), T) + 1;
        end
    end
    y = r(idx, :);
    h = lw_sharpe_hac_se(y);
    for j = 1:nb
        [ds, ss] = lw_sharpe_boot_draws(y, blocks(j), Minner, stream);
        ok = isfinite(ds) & isfinite(ss) & ss > 0;
        if ~any(ok) || ~isfinite(h.se), continue; end
        tabs = abs(ds(ok) - h.delta) ./ ss(ok);
        q = local_quantile_upper(tabs, 1 - alpha);
        evaluated(j) = evaluated(j) + 1;
        covered(j) = covered(j) + (abs(h.delta - target) <= q * h.se);
    end
end
cov = covered ./ evaluated;
[~, jbest] = min(abs(cov - (1 - alpha)));      % first minimum = smallest b
cal = struct('blocks', blocks, 'coverage_counts', covered, ...
    'evaluated_counts', evaluated, 'coverage', cov, 'block_size', blocks(jbest), ...
    'K', K, 'Minner', Minner, 'alpha', alpha, 'target_delta', target, ...
    'pseudo_model', 'stationary_bootstrap', 'pseudo_mean_block', pseudoMeanBlock);
end

function q = local_quantile_upper(x, p)
% Order statistic ceil(p*(n+1)), the convention of the +1 corrected p-value.
x = sort(x);
n = numel(x);
q = x(min(n, max(1, ceil(p * (n + 1)))));
end
