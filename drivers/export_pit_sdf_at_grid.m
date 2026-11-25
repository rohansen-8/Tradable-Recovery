function summary = export_pit_sdf_at_grid(inDir, outDir)
%% Evaluate the saved fitted h on the 500-point PIT grid supplied by the Python validation code.
%
% inDir holds the grids pitA_grid_<h>.csv written by the authors' Python
% validation code; for each date the saved empirical fit is evaluated at the
% grid points and the HJ-normalized SDF is formed as
%
%   M_t(m) = 1 - h_t(m,v_t) + qmean_t     (qmean_t as supplied with the grid)
%
% No Sharpe rescaling, no flooring, no clipping, no support truncation,
% no date dropping, no refit, no CV.

setpath();
maturities = {'quarterly','monthly','weekly','daily'};
if ~isfolder(outDir), mkdir(outDir); end
rows = cell(numel(maturities),1);

for im = 1:numel(maturities)
    mat = maturities{im};
    G = readtable(fullfile(inDir, sprintf('pitA_grid_%s.csv', mat)), ...
        'TextType','string','VariableNamingRule','preserve');

    cfg = config_empirical(mat);
    S = load(cfg.dataset.modelStateFile,'model_state'); ms = S.model_state;
    monLo = min(ms.mon_grid); monHi = max(ms.mon_grid);
    exoLo = min(ms.exo_grid); exoHi = max(ms.exo_grid);

    dates = unique(G.date,'stable');
    h_all = nan(height(G),1); M_all = nan(height(G),1);
    dRows = cell(numel(dates),1);

    for id = 1:numel(dates)
        idx = find(G.date == dates(id));
        m = G.m(idx); fq = G.f_Q(idx); v = G.vix(idx(1));
        qmean = G.qmean_t(idx(1));

        exo = build_exo_representer(cfg, ms, [], ...
            struct('mode','conditional','value',v));
        res = evaluate_sdf_grid(cfg, ms, m, exo);
        h = res.h(:);
        M = 1 - h + qmean;                  % HJ-normalized SDF (eq. hj_normalized_sdf)

        h_all(idx) = h; M_all(idx) = M;

        outside = (m < monLo) | (m > monHi);
        qMassOut = 0;
        % A single out-of-support point spans zero width, so it carries no mass;
        % guard also prevents trapz from reading a scalar 2nd arg as a dim index.
        if sum(outside) >= 2, qMassOut = trapz(m(outside), fq(outside)); end
        denom = trapz(m, fq ./ M);
        qmeanBond = trapz(m, h .* fq);      % diagnostic only

        dRows{id} = table(string(mat), dates(id), numel(m), min(m), max(m), ...
            monLo, monHi, sum(outside), mean(outside), qMassOut, trapz(m,fq), ...
            v, (v>=exoLo && v<=exoHi), min(M), max(M), sum(M<=0), ...
            all(isfinite(M)), denom, qmean, qmeanBond, qmean-qmeanBond, ...
            'VariableNames',{'maturity','date','n_grid','m_lo','m_hi', ...
            'mon_lo','mon_hi','n_out_of_support','frac_out_of_support', ...
            'q_mass_outside','q_mass_total','vix_t','vix_in_exo_range', ...
            'min_M','max_M','n_M_nonpositive','M_all_finite','denom_int_fQ_over_M', ...
            'qmean_frozen','qmean_bondarenko','qmean_diff'});
    end

    G.h_it = h_all; G.M_it = M_all;
    writetable(G, fullfile(outDir, sprintf('pitB_grid_%s.csv', mat)));
    D = cat(1, dRows{:});
    writetable(D, fullfile(outDir, sprintf('pitB_diagnostics_%s.csv', mat)));

    rows{im} = table(string(mat), height(D), sum(D.n_M_nonpositive), ...
        sum(D.n_M_nonpositive>0), min(D.min_M), max(D.max_M), ...
        mean(D.n_out_of_support>0), sum(D.q_mass_outside), ...
        median(D.q_mass_outside), max(D.q_mass_outside), ...
        sum(~D.vix_in_exo_range), all(D.M_all_finite), ...
        median(D.qmean_diff), min(D.qmean_diff), max(D.qmean_diff), ...
        mean(abs(D.qmean_diff)), monLo, monHi, ...
        'VariableNames',{'maturity','n_dates','n_grid_M_nonpositive', ...
        'n_dates_M_nonpositive','global_min_M','global_max_M', ...
        'frac_dates_endo_extrap','pooled_q_mass_outside','median_q_mass_outside', ...
        'max_q_mass_outside','n_dates_exo_extrap','all_M_finite', ...
        'median_qmean_diff','min_qmean_diff','max_qmean_diff', ...
        'mean_abs_qmean_diff','mon_lo','mon_hi'});
    fprintf('%-10s dates=%4d  min(M)=%10.6g  grid pts M<=0: %d  dates M<=0: %d  extrap dates: %d\n', ...
        mat, height(D), min(D.min_M), sum(D.n_M_nonpositive), ...
        sum(D.n_M_nonpositive>0), sum(D.n_out_of_support>0));
end

summary = cat(1, rows{:});
writetable(summary, fullfile(outDir,'pitB_summary.csv'));
fprintf('\nWrote Stage P-B to %s\n', outDir);

end
