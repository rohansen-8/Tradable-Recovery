function out = run_chj_hj_recovery_refit(lambdaMethod)
%% Recompute the CHJ SDF-recovery curves with the HJ-normalized SDF.
%
%   out = run_chj_hj_recovery_refit('eff_dim')
%
% CHJ counterpart of run_bs_hj_recovery_refit. The level of h is not
% identified by the option payoffs, so SDF recovery is measured with the
% HJ-normalized SDF. This driver refits the recovery datasets of the CHJ
% increasing-T study (the T = recovery_T
% seeds saved in chj_results.mat), with every replication-specific object
% (path, VIX lengthscale, exogenous grid, kernels) rebuilt exactly as in
% run_chj_simulated_shape_rejection_study, and rebuilds each curve with
% build_chj_sdf_recovery_curve, which uses the date-wise HJ-normalized SDF
% 1 - alpha_t(m)'h (eq. hj_normalized_sdf).
%
% Fits only: no CV, no shape tests, no parallel pool. Each refit must
% reproduce the saved lambda (eff_dim: re-solved by the rule and
% checked to 1e-10 relative; cv: the saved lambda is reused). Nothing
% existing is overwritten; the outputs are new files next to chj_results.mat:
%   chj_recovery_hj_raw.csv, chj_recovery_hj_summary.csv, chj_recovery_hj.mat

setpath();
if nargin < 1 || isempty(lambdaMethod)
    lambdaMethod = get_frozen_design().simulation_lambda_method;
end
lambdaMethod = char(lambdaMethod);
assert(ismember(lambdaMethod, {'eff_dim', 'cv'}), ...
    'run_chj_hj_recovery_refit:BadMethod', 'lambdaMethod must be ''eff_dim'' or ''cv''.');

cfg0 = config_chj_simulation_design(lambdaMethod);
Tval = cfg0.diagnostics.recovery_T;
simDir = fullfile(simulation_output_dir('simulations', lambdaMethod), 'chj_increasing_t');
resultFile = fullfile(simDir, 'chj_results.mat');
assert(isfile(resultFile), 'run_chj_hj_recovery_refit:NoResults', ...
    'Missing completed CHJ increasing-T results: %s', resultFile);

rawCsv = fullfile(simDir, 'chj_recovery_hj_raw.csv');
sumCsv = fullfile(simDir, 'chj_recovery_hj_summary.csv');
matFile = fullfile(simDir, 'chj_recovery_hj.mat');
existing = {rawCsv, sumCsv, matFile};
existing = existing(cellfun(@isfile, existing));
assert(isempty(existing), 'run_chj_hj_recovery_refit:Exists', ...
    'HJ recovery outputs already exist; refusing to overwrite: %s', strjoin(existing, ', '));

% Saved recovery datasets define the seeds, replication ids and lambdas.
S = load(resultFile, 'recovery_raw');
ref = S.recovery_raw(S.recovery_raw.Tval == Tval, {'rep', 'seed', 'lambda'});
ref = unique(ref, 'rows');
assert(numel(unique(ref.seed)) == height(ref), 'run_chj_hj_recovery_refit:Ambiguous', ...
    'Saved recovery rows carry more than one lambda per seed.');
ref = sortrows(ref, 'rep');
nRep = height(ref);

% T-specific objects, built exactly as in the simulation study (preByT).
cfgT = cfg0;
cfgT.sim.T = Tval;
mon_grid = linspace(cfgT.training.estimator_domain(1), ...
    cfgT.training.estimator_domain(2), cfgT.training.n_m_grid).';
option_m_grid = chj_option_moneyness_grid(cfgT);
theta_m = median_pairwise_lengthscale_weighted( ...
    option_m_grid, repmat(cfgT.sim.T, numel(option_m_grid), 1));
cfgT.kernel.endo.lengthscale = theta_m;
cfgT.training.mon_grid = mon_grid;
pre_base = build_chj_alpha_precompute(cfgT, mon_grid, []);

rowsCells = cell(nRep, 1);
lambdaRefit = NaN(nRep, 1);
relLambdaDiff = NaN(nRep, 1);
tStart = tic;
for i = 1:nRep
    seed = ref.seed(i);

    % Replication-specific objects, as in local_one_replication.
    cfg = cfgT;
    cfg.sim.T = Tval;
    cfg.sim.seed = seed;
    cfg.kernel.endo.lengthscale = theta_m;
    cfg.training.mon_grid = mon_grid;
    path = simulate_chj_numeric_path(cfg, seed);
    theta_v = median_pairwise_lengthscale_weighted( ...
        path.VIX, repmat(cfg.sim.J, Tval, 1));
    exo_grid = build_chj_exo_grid(cfg, path.VIX);
    cfg.kernel.exo.lengthscale = theta_v;
    cfg.training.exo_grid = exo_grid;
    cfg.training.n_exo_grid = numel(exo_grid);
    pre = pre_base;
    pre.fixed_exo_grid = exo_grid;
    pre.ls_exo = theta_v;

    alpha_mat = build_chj_alpha_numeric(cfg, mon_grid, seed, pre, path);
    Tobs = size(alpha_mat, 2);
    assert(Tobs == Tval, 'run_chj_hj_recovery_refit:BadT', 'Simulated T differs from recovery_T.');
    [K_h, ~, ~, basis] = build_kernel_norm_matrix(cfg, mon_grid, exo_grid);

    if strcmp(lambdaMethod, 'eff_dim')
        payoff = build_payoff_factor(alpha_mat);
        sel = select_lambda_effective_dimension(payoff, K_h, Tobs, cfg0.training.effdim_c);
        lambdaRefit(i) = sel.lambda;
        relLambdaDiff(i) = abs(sel.lambda - ref.lambda(i)) / ref.lambda(i);
        assert(relLambdaDiff(i) <= 1e-10, 'run_chj_hj_recovery_refit:LambdaMismatch', ...
            'seed %d: refit lambda %.17g differs from saved %.17g.', ...
            seed, sel.lambda, ref.lambda(i));
        h_star = solve_projection_qr(payoff, K_h, ref.lambda(i));
    else
        lambdaRefit(i) = ref.lambda(i);
        relLambdaDiff(i) = 0;
        Q_alpha = (alpha_mat * alpha_mat.') / Tobs;
        A = Q_alpha + ref.lambda(i) * K_h;
        A = 0.5 * (A + A.');
        Rc = chol(A);
        h_star = Rc \ (Rc.' \ mean(alpha_mat, 2));
    end

    ms = struct('H_star', reshape(h_star(:), basis.n + 1, basis.p), ...
        'lambda', ref.lambda(i), 'exo_grid', exo_grid(:), 'ls_exo', theta_v);
    rowsCells{i} = build_chj_sdf_recovery_curve(cfg, ms, path, Tval, ref.rep(i), seed, pre);
    if mod(i, 50) == 0
        fprintf('CHJ HJ recovery refit: %d/%d datasets (%.0f s)\n', i, nRep, toc(tStart));
    end
end
recovery_hj_raw = vertcat(rowsCells{:});

% Pointwise summary across datasets, same layout as the BS HJ summary.
m = unique(recovery_hj_raw.m);
summary = table();
for k = 1:numel(m)
    sel = recovery_hj_raw.m == m(k);
    d = recovery_hj_raw.difference(sel);
    qs = local_quantile7(d, [0.05 0.25 0.5 0.75 0.95]);
    summary = [summary; table(Tval, numel(d), m(k), mean(d), qs(3), qs(1), qs(2), ...
        qs(4), qs(5), mean(recovery_hj_raw.estimated_sdf(sel)), ...
        mean(recovery_hj_raw.true_sdf(sel)), 'VariableNames', {'Tval','B','m', ...
        'mean_difference','median_difference','p05_difference','p25_difference', ...
        'p75_difference','p95_difference','mean_estimated_sdf','mean_true_sdf'})]; %#ok<AGROW>
end

run_metadata = simulation_run_metadata(lambdaMethod);
provenance = struct('source_results', resultFile, 'recovery_T', Tval, ...
    'n_datasets', nRep, 'max_rel_lambda_diff', max(relLambdaDiff), ...
    'recovery_measure', ['hj_normalized: 1 - alpha_t(m)''h per date, mean-one on ' ...
    'the shape-test grid, averaged over dates; truth at the same state h_t'], ...
    'elapsed_seconds', toc(tStart));

local_atomic_writetable(recovery_hj_raw, rawCsv);
local_atomic_writetable(summary, sumCsv);
tmp = [tempname(simDir) '.mat'];
save(tmp, 'recovery_hj_raw', 'summary', 'run_metadata', 'provenance', 'lambdaRefit', '-v7.3');
movefile(tmp, matFile);

fprintf(['CHJ HJ recovery refit complete: %d datasets at T=%d, max relative lambda ' ...
    'difference %.2e, %.0f s.\n  %s\n  %s\n  %s\n'], nRep, Tval, ...
    max(relLambdaDiff), toc(tStart), rawCsv, sumCsv, matFile);
out = struct('raw_file', rawCsv, 'summary_file', sumCsv, 'mat_file', matFile, ...
    'provenance', provenance);

end

function q = local_quantile7(x, p)
%% Linear-interpolation sample quantiles (Hyndman-Fan type 7).

x = sort(x(:));
n = numel(x);
h = (n - 1) * p(:) + 1;
lo = floor(h);
hi = min(lo + 1, n);
q = x(lo) + (h - lo) .* (x(hi) - x(lo));
q = q(:).';

end

function local_atomic_writetable(tbl, file)
%% Write a table to a temporary file and move it into place.

[d, ~, ext] = fileparts(file);
tmp = [tempname(d) ext];
writetable(tbl, tmp);
movefile(tmp, file);

end
