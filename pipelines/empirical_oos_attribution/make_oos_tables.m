function R = make_oos_tables(opts)
%% Verify the saved OOS window files and build the pre-specified performance and attribution tables.
%
% R = make_oos_tables(opts)   (reads saved window files only; no fitting)
%
% For each maturity: assemble the primary window files, verify them (all
% scheduled windows present, OOS dates = schedule, one code/data/protocol/
% MATLAB fingerprint equal to the recorded provenance, payoff identities,
% timing), and compute summarize_oos_attribution. The 'python_reference' run
% and any sensitivity folder named in opts.sensitivity are summarized the
% same way. Tables are written as CSV to <outRoot>/tables/.
% Inference: Bartlett-HAC delta method (asymptotic); it is NOT the
% Ledoit-Wolf studentized bootstrap used for the manuscript's Panel C.
%
% opts: outRoot (default production root), horizons, sensitivity (cell of
% {horizon, folder name}), idTol (default 1e-10, relative).

if nargin < 1 || isempty(opts), opts = struct(); end
oos_setup();
horizons = get_opt(opts, 'horizons', {'quarterly','monthly','weekly','daily'});
sens = get_opt(opts, 'sensitivity', {});
idTol = get_opt(opts, 'idTol', 1e-10);
base = config_empirical_oos('monthly', opts);
tabDir = fullfile(base.paths.outRoot, 'tables');
if ~isfolder(tabDir), mkdir(tabDir); end

R = struct();
perf = {}; legRows = {}; regRows = {}; stateRows = {}; verRows = {}; dsrRows = {};
specs = [cellfun(@(h) {h, 'primary', struct()}, horizons, 'UniformOutput', false), ...
    cellfun(@(h) {h, 'reconciliation_python_protocol', struct('information_rule', 'python_reference')}, ...
    horizons, 'UniformOutput', false), ...
    cellfun(@(s) {s{1}, s{2}, struct()}, sens, 'UniformOutput', false)];
for j = 1:numel(specs)
    [hz, run, extra] = specs{j}{:};
    o = merge(opts, extra);
    oos = config_empirical_oos(hz, o);
    outDir = fullfile(oos.paths.outRoot, run, hz);
    if ~isfolder(fullfile(outDir, 'windows'))
        fprintf('skip %s/%s: no windows\n', run, hz); continue
    end
    o.outDir = outDir;
    [T, meta] = assemble_oos_attribution(hz, o);
    V = verify_run(T, meta, outDir, oos, idTol);
    S = summarize_oos_attribution(T, meta);
    key = matlab.lang.makeValidName([run '_' hz]);
    R.(key) = struct('T', T, 'meta', meta, 'verify', V, 'summary', S);
    verRows{end+1} = [table(string(run), string(hz), 'VariableNames', {'run','maturity'}), struct2table(V, 'AsArray', true)]; %#ok<AGROW>

    sr = S.sharpe; g = @(nm) sr(sr.series == nm, :);
    tot = g("total_impl"); fwd = g("forward_impl"); opt = g("options"); mkt = g("market_impl");
    d = S.delta_sharpe;
    dTF = d(d.comparison == "total_impl - forward_impl", :);
    dTM = d(d.comparison == "total_impl - market_impl", :);
    perf{end+1} = table(string(run), string(hz), S.n, T.oos_date(1), T.oos_date(end), ...
        tot.mean_ann, tot.SR_ann, tot.SR_se, fwd.SR_ann, fwd.SR_se, opt.SR_ann, opt.SR_se, ...
        mkt.SR_ann, mkt.SR_se, dTF.dSR_ann, dTF.se, dTM.dSR_ann, dTM.se, ...
        S.finding_B.SR_without_mid, S.hac_lag, ...
        'VariableNames', {'run','maturity','n','first','last','total_mean_ann','SR_total','se_total', ...
        'SR_forward','se_forward','SR_options','se_options','SR_market','se_market', ...
        'dSR_total_forward','se_dSR_tf','dSR_total_market','se_dSR_tm','SR_total_without_nearATM','hac_lag'}); %#ok<AGROW>
    L = S.legs; L.run = repmat(string(run), height(L), 1); L.maturity = repmat(string(hz), height(L), 1);
    legRows{end+1} = L(:, [end-1 end 1:end-2]); %#ok<AGROW>
    G = S.regressions; G.run = repmat(string(run), height(G), 1); G.maturity = repmat(string(hz), height(G), 1);
    regRows{end+1} = G(:, [end-1 end 1:end-2]); %#ok<AGROW>
    d.run = repmat(string(run), height(d), 1); d.maturity = repmat(string(hz), height(d), 1);
    d.p_one_sided = 0.5 * erfc(d.dSR_ann ./ d.se / sqrt(2));
    dsrRows{end+1} = d(:, [end-2 end-1 1:end-3 end]); %#ok<AGROW>
    A = S.finding_A; B = S.finding_B;
    stateRows{end+1} = table(string(run), string(hz), mean(T.fwd_position), std(T.fwd_position), ...
        A.corr_position_vix, A.slope_on_vix, A.t_slope, A.timing_share, ...
        B.mean_ann_by_tercile(1), B.mean_ann_by_tercile(2), B.mean_ann_by_tercile(3), ...
        B.n_by_tercile(1), B.n_by_tercile(2), B.n_by_tercile(3), B.corr_mid_forward, ...
        B.intercept_ann, B.intercept_t, B.loading_sq, B.loading_sq_t, B.var_ratio_without_mid, ...
        'VariableNames', {'run','maturity','fwd_pos_mean','fwd_pos_sd','corr_pos_vix','slope_pos_vix', ...
        't_slope','timing_share','nearATM_ann_lowVIX','nearATM_ann_midVIX','nearATM_ann_highVIX', ...
        'n_low','n_mid','n_high','corr_nearATM_forward','nearATM_alpha_ann','nearATM_alpha_t', ...
        'nearATM_loading_mkt_sq','nearATM_loading_t','var_ratio_without_nearATM'}); %#ok<AGROW>
end

R.performance = vertcat(perf{:});
R.legs = vertcat(legRows{:});
R.regressions = vertcat(regRows{:});
R.delta_sharpe = vertcat(dsrRows{:});
R.state = vertcat(stateRows{:});
R.verification = vertcat(verRows{:});
for f = {'performance','legs','regressions','delta_sharpe','state','verification'}
    writetable(R.(f{1}), fullfile(tabDir, ['oos_' f{1} '.csv']));
end
save(fullfile(tabDir, 'oos_tables.mat'), 'R');
fprintf('Tables written to %s\n', tabDir);
end

function V = verify_run(T, meta, outDir, oos, idTol)
%% Sample, provenance and identity checks on one assembled run.
runs = dir(fullfile(outDir, 'run_*.mat'));
M = load(fullfile(outDir, runs(end).name), 'runMeta'); M = M.runMeta;
panel = load_oos_panel(oos);
sched = build_oos_schedule(oos, panel);
ids = 1:sched.n_windows;
if isfield(M, 'window_ids'), ids = M.window_ids; end
V = struct();
V.n_obs = height(T);
V.n_scheduled = numel(ids);
V.dates_match_schedule = isequal(T.oos_date, [sched.windows(ids).oos_date].');
V.first_oos = T.oos_date(1);
V.last_oos = T.oos_date(end);
V.K = sched.K; V.T_panel = sched.T;
V.code_hash = string(meta.code_hash);
V.code_hash_matches_run = strcmp(meta.code_hash, M.provenance.code_hash);
V.code_clean_at_run = M.provenance.is_clean;
V.repo_heads = string(strjoin(M.provenance.repo_heads, ' | '));
V.data_sha256 = string(meta.data_sha256);
V.data_sha_matches_file = strcmp(meta.data_sha256, M.provenance.data_sha256);
V.matlab_version = string(M.provenance.matlab_version);
V.identity_max_rel = max(abs([T.identity1; T.identity2; T.identity3])) / max(abs(T.total));
V.identity_ok = V.identity_max_rel <= idTol;
cells = T{:, ~cellfun(@isempty, regexp(T.Properties.VariableNames, '^c\d_(put|call)$', 'once'))};
V.sum_check_max = max(abs(T.total - T.forward - sum(cells, 2)));
V.timing_ok = all(T.train_last < T.oos_date) && all(T.min_slack_days >= 0);
V.n_reused = sum(T.reused);
V.n_embargo_binds = sum(T.n_excluded_unrealized > 0);
V.lambda_min = min(T.lambda); V.lambda_max = max(T.lambda);
V.effdim_resid_max = max(abs(T.effdim_residual));
V.all_ok = V.n_obs == V.n_scheduled && V.dates_match_schedule && V.code_hash_matches_run ...
    && V.data_sha_matches_file && V.identity_ok && V.timing_ok && V.sum_check_max <= idTol * max(abs(T.total));
end

function o = merge(a, b)
%% Fields of b override a.
o = a;
for f = fieldnames(b).', o.(f{1}) = b.(f{1}); end
end

function v = get_opt(opts, name, default)
%% Return opts.(name) or a default.
if isfield(opts, name) && ~isempty(opts.(name)), v = opts.(name); else, v = default; end
end
