function [T, meta] = assemble_oos_attribution(horizon, opts)
%% Collect saved window files into one OOS table (reads saved files only).
%
% Checks before anything is written: every scheduled window is present
% (unless opts.allowPartial), OOS dates are unique and increasing, and all
% windows share one code hash, data hash and protocol hash. Writes
% <outDir>/<horizon>_oos_components.{csv,mat}. These files are derived
% deterministically from the window files and may be regenerated.
%
% Columns: window info and selected lambda; state (VIX, Delta, rf_hold,
% carry_hold); fwd_position; payoff total, forward and the ten strike-bucket
% x option-type cells c<b>_put / c<b>_call (labels in meta.bucket_labels);
% identity residuals; full-half-spread entry cost by cell.
% opts.outDir reads another run folder (e.g. a sensitivity run); completeness
% is then checked against the window ids recorded in its run metadata.

if nargin < 2 || isempty(opts), opts = struct(); end
oos = config_empirical_oos(horizon, opts);
if isfield(opts, 'outDir') && ~isempty(opts.outDir)
    oos.paths.outDir = opts.outDir;
    oos.paths.windows = fullfile(opts.outDir, 'windows');
end
files = dir(fullfile(oos.paths.windows, 'w*_*.mat'));
assert(~isempty(files), 'assemble_oos_attribution:NoWindows', 'No window files in %s.', oos.paths.windows);
files = sort({files.name});
n = numel(files);

rows = cell(n, 1);
fps = cell(n, 1);
for j = 1:n
    S = load(fullfile(oos.paths.windows, files{j}), 'W');
    W = S.W;
    fps{j} = W.fingerprint;
    r = struct();
    r.window_id = W.window.id;
    r.oos_date = W.window.oos_date;
    r.n_train = W.window.n_train;
    r.train_first = W.window.train_first;
    r.train_last = W.window.train_last;
    r.n_excluded_unrealized = W.window.n_excluded_unrealized;
    r.min_slack_days = W.window.min_slack_days;
    r.lambda = W.fit.lambda;
    r.effdim_target = W.fit.effdim_target;
    r.effdim_N = W.fit.effdim_N_at_lambda;
    r.effdim_residual = W.fit.effdim_residual;
    r.effdim_rank = W.fit.effdim_rank;
    r.ls_endo = W.fit.ls_endo;
    r.ls_exo = W.fit.ls_exo;
    r.exo_lo = W.fit.exo_range(1);
    r.exo_hi = W.fit.exo_range(2);
    r.vix = W.state.vix;
    r.Delta = W.state.Delta;
    r.rf_hold = W.state.rf_hold;
    r.carry_hold = W.state.carry_hold;
    r.holding_days = W.state.holding_days;
    r.realization_rule = string(W.state.realization_rule);
    r.fwd_position = W.comp.fwd_position;
    r.total = W.comp.total;
    r.forward = W.comp.forward;
    for b = 1:5
        r.(sprintf('c%d_put', b)) = W.comp.cells(b, 1);
        r.(sprintf('c%d_call', b)) = W.comp.cells(b, 2);
    end
    for b = 1:5
        r.(sprintf('cost%d_put', b)) = W.comp.cost_cells(b, 1);
        r.(sprintf('cost%d_call', b)) = W.comp.cost_cells(b, 2);
    end
    r.identity1 = W.comp.identity1_residual;
    r.identity2 = W.comp.identity2_residual;
    r.identity3 = W.comp.identity3_residual;
    r.reused = ~isempty(W.reused_from);
    r.fit_seconds = W.timing.fit_seconds;
    rows{j} = r;
    labels = W.comp.bucket_labels;
end
T = struct2table([rows{:}].');
T = sortrows(T, 'oos_date');

assert(numel(unique(T.oos_date)) == n && issorted(T.oos_date), ...
    'assemble_oos_attribution:Dates', 'Duplicate or unordered OOS dates.');
same = @(f) isscalar(unique(cellfun(@(x) x.(f), fps, 'UniformOutput', false)));
assert(same('code_hash') && same('data_sha256') && same('protocol_hash'), ...
    'assemble_oos_attribution:Mixed', 'Window files come from different code, data or protocol.');
ids = T.window_id;
runs = dir(fullfile(oos.paths.outDir, 'run_*.mat'));
R = load(fullfile(oos.paths.outDir, runs(end).name), 'runMeta');
expected = 1:R.runMeta.n_windows;
if isfield(R.runMeta, 'window_ids'), expected = R.runMeta.window_ids; end
complete = isequal(ids(:).', expected(:).');
if ~get_opt(opts, 'allowPartial', false)
    assert(n == R.runMeta.n_windows && complete, 'assemble_oos_attribution:Incomplete', ...
        '%d of %d windows present.', n, R.runMeta.n_windows);
end

meta = struct('horizon', horizon, 'role', oos.role, 'protocol', oos.protocol, ...
    'bucket_labels', {labels}, 'code_hash', fps{1}.code_hash, ...
    'data_sha256', fps{1}.data_sha256, 'protocol_hash', fps{1}.protocol_hash, ...
    'n_windows_present', n, 'n_windows_scheduled', R.runMeta.n_windows, ...
    'all_scheduled_present', complete, 'initial_window_vix', R.runMeta.initial_window_vix);
if get_opt(opts, 'write', true)
    writetable(T, fullfile(oos.paths.outDir, sprintf('%s_oos_components.csv', horizon)));
    save(fullfile(oos.paths.outDir, sprintf('%s_oos_components.mat', horizon)), 'T', 'meta');
end

end

function v = get_opt(opts, name, default)
%% Return opts.(name) or a default.
if isfield(opts, name) && ~isempty(opts.(name)), v = opts.(name); else, v = default; end
end
