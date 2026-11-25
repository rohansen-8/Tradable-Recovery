function summary = export_validation_sdf_at_nodes(stageADir, outDir)
%% Evaluate the saved fitted h at quadrature nodes supplied by the Python validation code.
%
% Reads the node files stageA_nodes_<h>.csv written by the authors' Python
% validation code, evaluates the saved empirical fitted model at
% those exact nodes (no interpolation, no clipping, no dropping), and forms
%
%   qmean_t = sum_i q_it h_it / sum_i q_it
%   M_it    = 1 - h_it + qmean_t
%
% consistent with eq:hj_normalized_sdf. No Sharpe rescaling. No fitting, no CV.

setpath();
maturities = {'quarterly','monthly','weekly','daily'};
if ~isfolder(outDir), mkdir(outDir); end
rows = cell(numel(maturities),1);

for im = 1:numel(maturities)
    mat = maturities{im};
    nodeFile = fullfile(stageADir, sprintf('stageA_nodes_%s.csv', mat));
    assert(isfile(nodeFile), 'Missing Stage-A nodes: %s', nodeFile);
    N = readtable(nodeFile, 'TextType','string', 'VariableNamingRule','preserve');

    cfg = config_empirical(mat);
    S = load(cfg.dataset.modelStateFile, 'model_state');
    ms = S.model_state;
    monLo = min(ms.mon_grid); monHi = max(ms.mon_grid);
    exoLo = min(ms.exo_grid); exoHi = max(ms.exo_grid);

    dates = unique(N.date, 'stable');
    assert(numel(dates) == ms.T || numel(dates) <= ms.T, ...
        '%s: more quadrature dates (%d) than fitted dates (%d).', mat, numel(dates), ms.T);

    h_all = nan(height(N),1); M_all = nan(height(N),1); qm_all = nan(height(N),1);
    dRows = cell(numel(dates),1);
    fRows = cell(numel(dates),1);
    okMask = false(height(N),1);

    for id = 1:numel(dates)
        idx = find(N.date == dates(id));
        x = N.x_it(idx); q = N.q_it(idx); v = N.vix_t(idx(1));

        % Explicit failure accounting: never repair, replace, or silently drop.
        if any(~isfinite(x)) || any(x <= 0)
            fRows{id} = table(string(mat), dates(id), "STAGE_B_FAILED", ...
                string(sprintf('non-positive/non-finite quadrature node (min x = %.6g)', min(x))), ...
                numel(x), min(x), min(q), ...
                'VariableNames', {'maturity','date','status','reason', ...
                'n_nodes','min_x','min_q'});
            continue
        end

        exo = build_exo_representer(cfg, ms, [], ...
            struct('mode','conditional','value',v));
        res = evaluate_sdf_grid(cfg, ms, x, exo);
        h = res.h(:);
        okMask(idx) = true;

        qmean = sum(q .* h) / sum(q);          % E^Q_t[h_t]
        M = 1 - h + qmean;                     % up to irrelevant positive scale

        h_all(idx) = h; M_all(idx) = M; qm_all(idx) = qmean;

        inDom = (x >= monLo) & (x <= monHi);
        % Self-normalized weights q/M (recorded, not applied to any moment)
        w = q ./ M;
        dRows{id} = table(string(mat), dates(id), numel(x), v, qmean, ...
            min(h), max(h), min(M), max(M), sum(M <= 0), all(isfinite(M)), ...
            sum(~inDom), min(x), max(x), (v >= exoLo && v <= exoHi), ...
            sum(q), min(q), all(isfinite(q)), all(q >= 0), sum(w), ...
            'VariableNames', {'maturity','date','n_nodes','vix_t','qmean_t', ...
            'min_h','max_h','min_M','max_M','n_M_nonpositive','M_all_finite', ...
            'n_nodes_out_of_domain','min_x','max_x','vix_in_exo_range', ...
            'sum_q','min_q','q_all_finite','q_all_nonneg','sum_q_over_M'});
    end

    N.h_it = h_all; N.qmean_t = qm_all; N.M_it = M_all;
    N.in_domain = (N.x_it >= monLo) & (N.x_it <= monHi);
    N.stage_b_ok = okMask;
    writetable(N, fullfile(outDir, sprintf('stageB_nodes_%s.csv', mat)));

    dRows = dRows(~cellfun(@isempty, dRows));
    D = cat(1, dRows{:});
    writetable(D, fullfile(outDir, sprintf('stageB_diagnostics_%s.csv', mat)));
    fRows = fRows(~cellfun(@isempty, fRows));
    F = cat(1, fRows{:});
    if ~isempty(F)
        writetable(F, fullfile(outDir, sprintf('stageB_failures_%s.csv', mat)));
    end

    % Identity checks on the self-normalized weights: E^P[1] = 1 and invariance
    % to a positive rescaling of M.
    idOK = find(okMask);
    if ~isempty(idOK)
        dOK = unique(N.date(idOK), 'stable');
        ep1 = zeros(numel(dOK),1); inv = zeros(numel(dOK),1);
        for k = 1:numel(dOK)
            j = idOK(N.date(idOK) == dOK(k));
            qk = N.q_it(j); Mk = N.M_it(j); gk = (N.x_it(j) - 1).^2;
            ep1(k) = sum(qk./Mk) / sum(qk./Mk);
            a = sum(gk.*qk./Mk) / sum(qk./Mk);
            b = sum(gk.*qk./(137.3*Mk)) / sum(qk./(137.3*Mk));
            inv(k) = abs(a - b);
        end
        fprintf('           E^P[1]-1 max = %.3e | scale-invariance max |dev| = %.3e\n', ...
            max(abs(ep1-1)), max(inv));
    end

    rows{im} = table(string(mat), height(D), height(N), ...
        min(D.min_M), median(D.min_M), max(D.max_M), ...
        sum(D.n_M_nonpositive), sum(D.n_M_nonpositive > 0), ...
        mean(D.n_M_nonpositive > 0), sum(D.n_nodes_out_of_domain), ...
        sum(D.n_nodes_out_of_domain > 0), sum(~D.vix_in_exo_range), ...
        all(D.M_all_finite), all(D.q_all_finite), all(D.q_all_nonneg), ...
        max(abs(D.sum_q - 1)), monLo, monHi, exoLo, exoHi, ms.lambda, ms.T, ...
        'VariableNames', {'maturity','n_dates','n_nodes','global_min_M', ...
        'median_of_date_min_M','global_max_M','n_nodes_M_nonpositive', ...
        'n_dates_with_nonpositive_M','frac_dates_with_nonpositive_M', ...
        'n_nodes_out_of_domain','n_dates_with_out_of_domain', ...
        'n_dates_vix_out_of_exo_range','all_M_finite','all_q_finite', ...
        'all_q_nonneg','max_abs_sum_q_minus_1','mon_lo','mon_hi', ...
        'exo_lo','exo_hi','lambda','T_fitted'});
    fprintf('%-10s dates=%4d nodes=%5d  min(M)=%12.6g  nodes M<=0: %d  out-of-domain: %d\n', ...
        mat, height(D), height(N), min(D.min_M), sum(D.n_M_nonpositive), ...
        sum(D.n_nodes_out_of_domain));
end

summary = cat(1, rows{:});
writetable(summary, fullfile(outDir, 'stageB_summary.csv'));
fprintf('\nWrote Stage-B outputs to %s\n', outDir);

end
