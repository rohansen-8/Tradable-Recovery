function out = run_noncrisis_effdim_fits(horizons)
%% In-sample effective-dimension fits on the non-crisis windows (Table 4.1, Panel B).
%
%   out = run_noncrisis_effdim_fits()                 % all four maturities
%   out = run_noncrisis_effdim_fits({'monthly'})
%
% Non-crisis windows: the estimation panel is restricted to
% start <= date <= end and the full-sample pipeline is run on those rows.
%   quarterly, monthly: 2009-06-01 .. 2019-06-02
%   weekly:             2012-09-01 .. 2019-09-01
%   daily:              2016-06-01 .. 2019-06-02
% The estimator is the full-sample one (train_projection_model, eff_dim rule
% N_T(lambda) = T^(1/3) on the window's T):
%   * length scales are option-level medians of the window rows (computed
%     inside train_projection_model from the rows passed);
%   * the moneyness basis range and the fixed VIX range [5, 65] are the
%     configured constants; when the maturity's VIX range is data driven
%     (weekly), it is the min/max of VIX over the window rows.
% Only save paths are redirected (never results/empirical/), and cfg.real is
% removed so build_real_model_meta does not re-read the full panel.
% Outputs, refusing to overwrite: results/empirical_noncrisis/eff_dim/<h>/
%   <h>_noncrisis_model_state.mat, <h>_noncrisis_returns.csv,
%   <h>_noncrisis_metadata.mat.
% Strategy return: alpha_t' h_star (payoff conventions of build_alpha_panel,
% as in Panel A). Benchmark: the variance-swap series of the non-crisis
% return pairs (data/return_pairs), which does not depend on the estimator; dates are checked one-to-one.

if nargin < 1 || isempty(horizons)
    horizons = {'quarterly','monthly','weekly','daily'};
end
setpath();
root = fileparts(fileparts(mfilename('fullpath')));
win = struct('quarterly', {{'2009-06-01','2019-06-02'}}, 'monthly', {{'2009-06-01','2019-06-02'}}, ...
    'weekly', {{'2012-09-01','2019-09-01'}}, 'daily', {{'2016-06-01','2019-06-02'}});
ann = struct('quarterly', 4, 'monthly', 12, 'weekly', 52, 'daily', 252);
rp = fullfile(root, 'data', 'return_pairs');   % variance-swap benchmark series (licensed-derived, local)
rows = {};
for i = 1:numel(horizons)
    h = horizons{i};
    outDir = fullfile(results_root(), 'empirical_noncrisis', 'eff_dim', h);
    msFile = fullfile(outDir, [h '_noncrisis_model_state.mat']);
    retFile = fullfile(outDir, [h '_noncrisis_returns.csv']);
    metaFile = fullfile(outDir, [h '_noncrisis_metadata.mat']);
    assert(~isfile(msFile) && ~isfile(retFile) && ~isfile(metaFile), ...
        'run_noncrisis_effdim_fits:Exists', 'Outputs for %s exist; refusing to overwrite.', h);
    if ~isfolder(outDir), mkdir(outDir); end
    scratch = fullfile(outDir, '_scratch');
    if ~isfolder(scratch), mkdir(scratch); end

    cfg = config_empirical(h, 'eff_dim');
    df = readtable(cfg.dataset.dfFile, 'VariableNamingRule', 'preserve');
    if ~isdatetime(df.date), df.date = datetime(string(df.date), 'InputFormat', 'yyyy-MM-dd'); end
    w = win.(h);
    keep = df.date >= datetime(w{1}) & df.date <= datetime(w{2});
    dfW = df(keep, :);

    cfgW = cfg;
    if isfield(cfgW, 'real'), cfgW = rmfield(cfgW, 'real'); end
    cfgW.dataset.path = scratch;
    cfgW.dataset.modelStateFile = fullfile(scratch, [h '_noncrisis_scratch.mat']);
    cfgW.results.path = scratch; cfgW.results.methodPath = scratch; cfgW.results.figuresPath = scratch;
    cfgW.results.cvCheckpoint = fullfile(scratch, 'unused_cv.mat');
    protected = [fullfile(results_root(), 'empirical') filesep];
    assert(~startsWith(cfgW.dataset.modelStateFile, protected), 'protected save path');
    m = get_empirical_maturity_config(h);
    if m.exo_range_is_data_driven
        v = dfW.VIX;
        cfgW.training.exo_domain = [min(v), max(v)];
        cfgW.training.exo_grid = linspace(min(v), max(v), cfgW.training.nExo).';
    end

    model_state = train_projection_model(cfgW, dfW);
    assert(strcmp(char(string(model_state.lambda_method)), 'eff_dim'), 'not an eff_dim fit');
    s = model_state.alpha.' * model_state.h_star;
    dates = unique(dfW.date);
    assert(numel(s) == numel(dates) && model_state.T == numel(dates), 'T mismatch');

    P = readtable(fullfile(rp, ['noncrisis_is_' h '_return_pairs.csv']));
    dP = P.date; if ~isdatetime(dP), dP = datetime(string(dP)); end
    assert(height(P) == numel(dates) && all(dateshift(dP, 'start', 'day') == dates), ...
        'run_noncrisis_effdim_fits:Dates', '%s: benchmark dates differ from the window dates.', h);
    R = table(string(dates, 'yyyy-MM-dd'), s(:), P.vswap_return, ...
        'VariableNames', {'date','strategy_return','vswap_return'});
    srS = mean(s) / std(s) * sqrt(ann.(h));
    srV = mean(P.vswap_return) / std(P.vswap_return) * sqrt(ann.(h));

    save(msFile, 'model_state', '-v7.3');
    writetable(R, retFile);
    [~, head] = system(sprintf('git -C "%s" rev-parse HEAD', root));
    [~, st] = system(sprintf('git -C "%s" status --porcelain -- training evaluation configs kernels drivers', root));
    est = [dir(fullfile(root, 'training', '*.m')); dir(fullfile(root, 'drivers', 'run_noncrisis_effdim_fits.m'))];
    hashes = strings(numel(est), 1);
    for k = 1:numel(est), hashes(k) = string(est(k).name) + " " + local_sha256(fullfile(est(k).folder, est(k).name)); end
    meta = struct('horizon', h, 'window', {w}, 'T', model_state.T, 'n_rows', height(dfW), ...
        'lambda', model_state.lambda, 'effdim_target', model_state.T^(1/3), ...
        'exo_domain', cfgW.training.exo_domain, 'mon_domain', cfgW.training.mon_domain, ...
        'sr_strategy_annual', srS, 'sr_vswap_annual', srV, 'annualization', ann.(h), ...
        'panel_file', cfg.dataset.dfFile, 'panel_sha256', local_sha256(cfg.dataset.dfFile), ...
        'benchmark_file', fullfile(rp, ['noncrisis_is_' h '_return_pairs.csv']), ...
        'git_head', strtrim(head), 'git_status', string(strtrim(st)), 'code_sha256', hashes, ...
        'matlab_version', version, 'created', string(datetime('now')));
    save(metaFile, 'meta');
    rmdir(scratch, 's');
    rows(end+1, :) = {h, model_state.T, model_state.lambda, model_state.T^(1/3), srS, srV}; %#ok<AGROW>
    fprintf('%-9s T=%4d lambda=%.6g SR=%.4f VS=%.4f\n', h, model_state.T, model_state.lambda, srS, srV);
end
out = cell2table(rows, 'VariableNames', {'horizon','T','lambda','effdim_target','sr_strategy','sr_vswap'});
end

function hx = local_sha256(f)
md = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(f, 'r'); b = fread(fid, inf, '*uint8'); fclose(fid);
md.update(b);
hx = lower(reshape(dec2hex(typecast(md.digest(), 'uint8')).', 1, []));
end
