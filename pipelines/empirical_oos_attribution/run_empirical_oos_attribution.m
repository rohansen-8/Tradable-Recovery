function out = run_empirical_oos_attribution(horizons, opts)
%% Expanding-window OOS fits and payoff attribution, checkpointed per window.
%
% out = run_empirical_oos_attribution({'monthly'}, opts)
%
% opts fields (all optional):
%   information_rule  'realized_payoff' (default, primary) | 'python_reference'
%   mode              'validation' (default) | 'production'
%   outRoot           output root; validation runs must NOT use the default
%                     production root, production runs must use it
%   windows           window ids to run (default: all)
%   maxNewFits        stop cleanly after this many new fits (default Inf)
%   sdfRoot           canonical SDF checkout (oos_setup default)
%
% Each OOS window is fitted once and written atomically to
%   <outDir>/windows/w####_yyyymmdd.mat
% with a fingerprint (protocol, code hash, data hash, MATLAB version,
% training-set hash).
% On rerun a window file with a matching fingerprint is loaded, never refit;
% a mismatching file is an error, never overwritten. A 'python_reference' run
% reuses the primary window file whenever the training set is identical, so
% no fit is repeated.
% Production gate: every code dependency must be tracked and unmodified in Git
% (oos_provenance.is_clean) and output must go to the default root.

if nargin < 1 || isempty(horizons), horizons = {'monthly','weekly','daily','quarterly'}; end
if nargin < 2 || isempty(opts), opts = struct(); end
horizons = cellstr(horizons);
oos_setup(get_opt(opts, 'sdfRoot', ''));
mode = validatestring(char(get_opt(opts, 'mode', 'validation')), {'validation','production'});
maxNewFits = get_opt(opts, 'maxNewFits', Inf);

out = struct();
for h = 1:numel(horizons)
    oos = config_empirical_oos(horizons{h}, opts);
    if strcmp(mode, 'production')
        assert(oos.paths.is_default_root, 'run_empirical_oos_attribution:Root', ...
            'Production output must use the default root.');
    else
        assert(~oos.paths.is_default_root, 'run_empirical_oos_attribution:Root', ...
            'Validation runs may not write to the production OOS root; pass opts.outRoot.');
    end
    prov = oos_provenance(oos);
    if strcmp(mode, 'production')
        assert(prov.is_clean, 'run_empirical_oos_attribution:DirtyCode', ...
            'Production requires committed, unmodified code. Dirty: %s', strjoin(prov.dirty_files, ', '));
    end
    panel = load_oos_panel(oos);
    sched = build_oos_schedule(oos, panel);
    ensure_dir(oos.paths.windows);
    ensure_dir(oos.paths.scratch);

    protocolHash = oos_sha256([oos.horizon '|' jsonencode(oos.protocol)]);
    core = rmfield(oos.protocol, 'information_rule');
    coreHash = oos_sha256([oos.horizon '|' jsonencode(core)]);

    runMeta = struct('mode', mode, 'protocol', oos.protocol, 'role', oos.role, ...
        'protocol_hash', protocolHash, 'provenance', prov, 'K', sched.K, 'T', sched.T, ...
        'n_windows', sched.n_windows, 'n_windows_embargo_binds', sched.n_windows_embargo_binds, ...
        'thin_dates', panel.thin_dates, 'n_dates_unfiltered', panel.n_dates_unfiltered, ...
        'initial_window_vix', panel.dates_table.vix(sched.windows(1).train_index), ...
        'paths', oos.paths);
    save(fullfile(oos.paths.outDir, sprintf('run_%s.mat', ...
        char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS')))), 'runMeta');

    sel = get_opt(opts, 'windows', 1:sched.n_windows);
    st = struct('n_fit', 0, 'n_loaded', 0, 'n_reused', 0, 'complete', true);
    for k = sel(:).'
        win = sched.windows(k);
        fp = struct('horizon', oos.horizon, 'window_id', win.id, ...
            'oos_date', char(string(win.oos_date, 'yyyy-MM-dd')), ...
            'protocol_hash', protocolHash, 'protocol_core_hash', coreHash, ...
            'code_hash', prov.code_hash, 'data_sha256', prov.data_sha256, ...
            'matlab_version', prov.matlab_version, 'train_hash', oos_sha256(win.train_key));
        fname = sprintf('w%04d_%s.mat', win.id, char(string(win.oos_date, 'yyyyMMdd')));
        target = fullfile(oos.paths.windows, fname);

        if isfile(target)
            S = load(target, 'W');
            assert(isequal(S.W.fingerprint, fp), 'run_empirical_oos_attribution:Fingerprint', ...
                'Existing %s has a different fingerprint; refusing to reuse or overwrite.', target);
            st.n_loaded = st.n_loaded + 1;
            continue
        end

        primaryFile = fullfile(oos.paths.primaryWindows, fname);
        if strcmp(oos.protocol.information_rule, 'python_reference') && isfile(primaryFile)
            S = load(primaryFile, 'W');
            pf = S.W.fingerprint;
            if strcmp(pf.train_hash, fp.train_hash) && strcmp(pf.protocol_core_hash, coreHash) ...
                    && strcmp(pf.code_hash, fp.code_hash) && strcmp(pf.data_sha256, fp.data_sha256) ...
                    && strcmp(pf.matlab_version, fp.matlab_version)
                W = S.W;
                W.fingerprint = fp;
                W.reused_from = primaryFile;
                atomic_save(target, W);
                st.n_reused = st.n_reused + 1;
                continue
            end
        end

        if st.n_fit >= maxNewFits
            st.complete = false;
            break
        end
        W = compute_window(oos, panel, win, fp);
        atomic_save(target, W);
        st.n_fit = st.n_fit + 1;
    end
    st.windows_dir = oos.paths.windows;
    st.n_windows = sched.n_windows;
    out.(oos.horizon) = st;
    fprintf('%s [%s]: fitted %d, loaded %d, reused %d, complete=%d\n', oos.horizon, ...
        oos.protocol.information_rule, st.n_fit, st.n_loaded, st.n_reused, st.complete);
end

end

function W = compute_window(oos, panel, win, fp)
%% Fit one window once and evaluate every payoff component from that fit.
t0 = tic;
ms = fit_oos_window(oos, panel, win);
tFit = toc(t0);
rows = panel.df.date == win.oos_date;
dfTau = panel.df(rows, :);
t1 = tic;
comp = evaluate_oos_payoff_components(oos, ms, dfTau, panel.half_spread(rows));
tEval = toc(t1);
D = panel.dates_table(win.oos_index, :);
trainDates = panel.dates(win.train_index);

W = struct();
W.fingerprint = fp;
W.window = struct('id', win.id, 'oos_date', win.oos_date, 'n_train', win.n_train, ...
    'train_first', trainDates(1), 'train_last', trainDates(end), ...
    'n_excluded_unrealized', win.n_excluded_unrealized, 'min_slack_days', win.min_slack_days);
W.fit = struct('lambda', ms.lambda, 'T', ms.T, 'h_star', ms.h_star, ...
    'effdim_target', ms.effdim.target, 'effdim_N_at_lambda', ms.effdim.N_at_lambda, ...
    'effdim_residual', ms.effdim.residual, 'effdim_rank', ms.effdim.numerical_rank, ...
    'effdim_rank_upper_bound', ms.effdim.rank_upper_bound, 'ls_endo', ms.ls_endo, ...
    'ls_exo', ms.ls_exo, 'exo_range', [min(ms.exo_grid(:)) max(ms.exo_grid(:))], ...
    'solver_status', ms.solver_status, 'condition_number', ms.condition_number);
W.comp = comp;
W.state = struct('vix', D.vix, 'Delta', D.Delta, 'S', D.S, 'S_next', D.S_next, ...
    'rf_hold', D.rf_hold, 'carry_hold', D.carry_hold, 'holding_days', D.holding_days, ...
    'realization_rule', D.realization_rule, 'n_expiries', D.n_expiries);
W.reused_from = '';
W.timing = struct('fit_seconds', tFit, 'eval_seconds', tEval);
W.created_on = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
end

function atomic_save(target, W)
%% Write to a temporary file, then rename; never replace an existing file.
assert(~isfile(target), 'run_empirical_oos_attribution:Exists', '%s already exists.', target);
tmp = [target '.tmp_' char(java.util.UUID.randomUUID())];
save(tmp, 'W', '-v7');
[ok, msg] = movefile(tmp, target);
assert(ok, 'run_empirical_oos_attribution:Move', 'Could not move %s: %s', tmp, msg);
end

function ensure_dir(d)
%% Create a folder if needed.
if ~isfolder(d), mkdir(d); end
end

function v = get_opt(opts, name, default)
%% Return opts.(name) or a default.
if isfield(opts, name) && ~isempty(opts.(name)), v = opts.(name); else, v = default; end
end
