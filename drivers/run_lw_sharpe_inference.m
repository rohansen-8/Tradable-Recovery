function out = run_lw_sharpe_inference(mode, cells)
%% Ledoit-Wolf Sharpe-difference inference for the strategy versus the variance swap.
%
%   out = run_lw_sharpe_inference('validate_cv')   % check against 11 stored CV reference cells
%   out = run_lw_sharpe_inference('eff_dim')       % effective-dimension full-sample and OOS cells
%   out = run_lw_sharpe_inference(mode, cells)     % subset, e.g. {'oos_monthly'}
%
% validate_cv: reruns the full procedure (block calibration and final
% bootstrap) on the stored paired returns of the CV-based fits and compares
% with the stored reference realizations (tests/fixtures/ledoit_wolf_cv): selected
% block, coverage counts, two-sample KS on d_star, se_star and
% delta_star, and the one-sided p-value with its Monte Carlo error.
%
% eff_dim: paired returns are rebuilt from saved outputs only:
%   full_is_<h>: alpha_t'h of results/empirical/eff_dim/<h>/<h>_model_state.mat
%                against the variance-swap series of the return pairs;
%   noncrisis_is_<h>: results/empirical_noncrisis/eff_dim/<h>/<h>_noncrisis_returns.csv
%                (run_noncrisis_effdim_fits).
%   oos_<h>:     column total of results/empirical_oos/eff_dim/primary/<h>/
%                <h>_oos_components.csv against the OOS variance-swap series.
% The non-crisis cells require run_noncrisis_effdim_fits; any subset of
% cells can be run by name.
%
% Outputs go to results/inference/performance_sharpe/<mode>_<calTag>/ (refuses to
% overwrite): one .mat per cell, the paired returns used, a summary CSV,
% and run metadata with the SHA-256 of the package files and inputs.

setpath();
root = fileparts(fileparts(mfilename('fullpath')));
rp = fullfile(root, 'data', 'return_pairs');   % variance-swap benchmark series (licensed-derived, local)
fx = fullfile(root, 'tests', 'fixtures', 'ledoit_wolf_cv');
ann = struct('quarterly', 4, 'monthly', 12, 'weekly', 52, 'daily', 252);
mode = char(mode);
switch mode
    case 'validate_cv'
        allCells = {'full_is_quarterly','full_is_monthly','full_is_weekly','full_is_daily', ...
            'noncrisis_is_quarterly','noncrisis_is_monthly','noncrisis_is_weekly', ...
            'noncrisis_is_daily','oos_monthly','oos_weekly','oos_daily'};
    case 'eff_dim'
        allCells = {'full_is_quarterly','full_is_monthly','full_is_weekly','full_is_daily', ...
            'noncrisis_is_quarterly','noncrisis_is_monthly','noncrisis_is_weekly', ...
            'noncrisis_is_daily','oos_monthly','oos_weekly','oos_daily'};
    otherwise
        error('run_lw_sharpe_inference:BadMode', 'mode must be validate_cv or eff_dim.');
end
if nargin < 2 || isempty(cells), cells = allCells; end
assert(all(ismember(cells, allCells)), 'run_lw_sharpe_inference:BadCell', 'unknown cell');

% Calibration-model tag: block-size calibration with stationary-bootstrap
% pseudo data, mean block length 10.
calTag = 'sb10';
outDir = fullfile(results_root(), 'inference', 'performance_sharpe', [mode '_' calTag]);
if ~isfolder(outDir), mkdir(outDir); end
meta = local_metadata(root, mode);
rows = cell(numel(cells), 1);
for i = 1:numel(cells)
    c = cells{i};
    cellFile = fullfile(outDir, [c '.mat']);
    assert(~isfile(cellFile), 'run_lw_sharpe_inference:Exists', ...
        'Refusing to overwrite %s', cellFile);
    parts = strsplit(c, '_'); h = parts{end};
    tic;
    switch mode
        case 'validate_cv'
            z = load(fullfile(fx, [c '.mat']));
            pairs = table((1:size(z.r,1)).', z.r(:,1), z.r(:,2), ...
                'VariableNames', {'t','strategy_return','vswap_return'});
        case 'eff_dim'
            pairs = local_effdim_pairs(root, rp, c, h);
    end
    r = [pairs.strategy_return, pairs.vswap_return];
    res = lw_sharpe_difference_test(r, ann.(h));
    res.cell = c; res.mode = mode; res.elapsed_seconds = toc;
    row = table(string(c), res.T, res.sr_strategy_annual, res.sr_benchmark_annual, ...
        res.delta_sr_annual, res.d_observed, res.block_size, res.n_valid, ...
        res.p_value_one_sided, res.p_value_two_sided, string(local_stars(res.p_value_one_sided)), ...
        'VariableNames', {'cell','T','sr_strategy','sr_vswap','delta_sr','d_observed', ...
        'block_size','n_valid','p_one_sided','p_two_sided','stars'});
    if strcmp(mode, 'validate_cv')
        [row, res.validation] = local_validate(row, res, z);
    end
    writetable(pairs, fullfile(outDir, [c '_return_pairs.csv']));
    tmp = [tempname(outDir) '.mat'];
    save(tmp, 'res', 'meta', '-v7.3');
    movefile(tmp, cellFile);
    rows{i} = row;
    fprintf('%-24s T=%5d  dSR=%6.3f  block=%2d  p1=%.4f %s  (%.0f s)\n', c, res.T, ...
        res.delta_sr_annual, res.block_size, res.p_value_one_sided, ...
        local_stars(res.p_value_one_sided), res.elapsed_seconds);
end
summary = vertcat(rows{:});
sumFile = fullfile(outDir, sprintf('summary_%s.csv', char(datetime('now','Format','yyyyMMdd_HHmmss'))));
writetable(summary, sumFile);
save(fullfile(outDir, sprintf('run_metadata_%s.mat', char(datetime('now','Format','yyyyMMdd_HHmmss')))), 'meta');
fprintf('Saved %s\n', sumFile);
out = struct('summary', summary, 'out_dir', outDir, 'meta', meta);
end

function pairs = local_effdim_pairs(root, rp, c, h)
if startsWith(c, 'full_is_')
    S = load(fullfile(results_root(), 'empirical', 'eff_dim', h, [h '_model_state.mat']), 'model_state');
    ms = S.model_state;
    assert(strcmp(char(string(ms.lambda_method)), 'eff_dim'), 'not an eff_dim fit');
    s = ms.alpha.' * ms.h_star;
    P = readtable(fullfile(rp, ['full_is_' h '_return_pairs.csv']));
    assert(height(P) == numel(s), 'full-sample length mismatch');
    pairs = table(string(P.date), s(:), P.vswap_return, 'VariableNames', ...
        {'date','strategy_return','vswap_return'});
elseif startsWith(c, 'noncrisis_is_')
    % In-sample eff_dim fits on the non-crisis windows (run_noncrisis_effdim_fits).
    R = readtable(fullfile(results_root(), 'empirical_noncrisis', 'eff_dim', h, ...
        [h '_noncrisis_returns.csv']), 'TextType', 'string');
    pairs = table(string(R.date), R.strategy_return, R.vswap_return, 'VariableNames', ...
        {'date','strategy_return','vswap_return'});
else
    O = readtable(fullfile(results_root(), 'empirical_oos', 'eff_dim', 'primary', h, ...
        [h '_oos_components.csv']));
    P = readtable(fullfile(rp, ['oos_' h '_return_pairs.csv']));
    dO = datetime(string(O.oos_date), 'InputFormat', 'yyyy-MM-dd');
    dP = P.date; if ~isdatetime(dP), dP = datetime(string(dP), 'InputFormat', 'dd-MMM-yyyy', 'Locale', 'en_US'); end
    assert(height(O) == height(P) && all(dO == dateshift(dP, 'start', 'day')), 'OOS dates do not align');
    pairs = table(string(dO, 'yyyy-MM-dd'), O.total, P.vswap_return, 'VariableNames', ...
        {'date','strategy_return','vswap_return'});
end
end

function [row, v] = local_validate(row, res, z)
ok = logical(z.valid(:));
ksd = local_ks(abs((res.delta_star - res.delta_sr_period) ./ res.se_star), z.d_star(ok));
kss = local_ks(res.se_star, z.se_star(ok));
ksl = local_ks(res.delta_star, z.delta_star(ok));
p0 = (1 + sum((z.delta_star(ok) - z.delta_sr_period) ./ z.se_star(ok) >= z.d_observed)) / (sum(ok) + 1);
mcse = sqrt(max(p0 * (1 - p0), 1 / numel(ok)) * 2 / numel(ok));
v = struct('block_preserved', double(z.block_size), 'block_match', double(z.block_size) == res.block_size, ...
    'coverage_preserved', double(z.coverage_counts(:)).', 'coverage_rebuilt', res.calibration.coverage_counts, ...
    'ks_p_dstar', ksd, 'ks_p_sestar', kss, 'ks_p_deltastar', ksl, ...
    'p_preserved', p0, 'p_diff', res.p_value_one_sided - p0, 'p_mcse_diff', mcse, ...
    'stars_preserved', string(local_stars(p0)));
row.block_preserved = v.block_preserved;
row.p_preserved = p0;
row.p_diff_over_mcse = v.p_diff / mcse;
row.stars_preserved = v.stars_preserved;
row.ks_p_dstar = ksd; row.ks_p_sestar = kss; row.ks_p_deltastar = ksl;
row.max_abs_coverage_diff = max(abs(double(z.coverage_counts(:)).' - res.calibration.coverage_counts));
end

function p = local_ks(x, y)
% Two-sample Kolmogorov-Smirnov p-value (asymptotic), base MATLAB.
x = sort(x(:)); y = sort(y(:)); n = numel(x); m = numel(y);
v = unique([x; y]);
D = max(abs(local_ecdf(x, v) - local_ecdf(y, v)));
ne = n * m / (n + m);
lam = (sqrt(ne) + 0.12 + 0.11 / sqrt(ne)) * D;
j = (1:100).';
p = min(1, max(0, 2 * sum((-1).^(j - 1) .* exp(-2 * j.^2 * lam^2))));
end

function F = local_ecdf(x, v)
F = arrayfun(@(t) sum(x <= t), v) / numel(x);
end

function s = local_stars(p)
if p < 0.01, s = '***'; elseif p < 0.05, s = '**'; elseif p < 0.10, s = '*'; else, s = ''; end
end

function meta = local_metadata(root, mode)
files = [dir(fullfile(root, 'testing', 'lw_sharpe_*.m')); dir(fullfile(root, 'drivers', 'run_lw_sharpe_inference.m'))];
hashes = strings(numel(files), 1);
for i = 1:numel(files)
    hashes(i) = string(files(i).name) + " " + local_sha256(fullfile(files(i).folder, files(i).name));
end
[~, head] = system(sprintf('git -C "%s" rev-parse HEAD', root));
[~, st] = system(sprintf('git -C "%s" status --porcelain -- testing drivers tests', root));
meta = struct('mode', mode, 'created', string(datetime('now')), 'git_head', strtrim(head), ...
    'git_status_code_dirs', string(strtrim(st)), 'package_sha256', hashes, ...
    'matlab_version', version, 'procedure', ['Ledoit-Wolf (2008) studentized circular-block ' ...
    'bootstrap, rebuilt 2026-11-21; HAC: prewhitened QS (gate 1 exact on 11 preserved cells); ' ...
    'calibration: Algorithm 3.1 with stationary-bootstrap pseudo data, mean block 10 (reconstruction).']);
end

function h = local_sha256(f)
md = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(f, 'r'); b = fread(fid, inf, '*uint8'); fclose(fid);
md.update(b);
d = typecast(md.digest(), 'uint8');
h = lower(reshape(dec2hex(d).', 1, []));
end
