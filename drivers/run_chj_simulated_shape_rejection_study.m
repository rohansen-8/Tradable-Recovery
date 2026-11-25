function [summary, raw, regularization_bias_raw, recovery_raw] = run_chj_simulated_shape_rejection_study(Tvals, B, lambda, seed0, nMC, tag, saveOutput, useParallel, cov_opts, designB, lambdaMethod)
%% Run CHJ shape-rejection simulations under the pre-specified study conventions.
%
% Each independent dataset is simulated once. Its moneyness and VIX kernel
% lengthscales follow the empirical median-pairwise rule, fitting lambda is selected by
% shape-blind one-step-ahead projection-loss CV (lambdaMethod 'cv', default)
% or by the effective-dimension rule (lambdaMethod 'eff_dim') unless
% explicitly supplied, and the final tests use clipped equal-date VIX
% averaging with Bartlett HAC.

setpath();
if nargin < 11 || isempty(lambdaMethod), lambdaMethod = 'cv'; end
cfg0 = config_chj_simulation_design(lambdaMethod);

if nargin < 1 || isempty(Tvals), Tvals = cfg0.sim.performance_T(1); end
if nargin < 2 || isempty(B), B = 10; end
if nargin < 3, lambda = cfg0.training.lambda; end
if nargin < 4 || isempty(seed0), seed0 = cfg0.sim.seed0; end
if nargin < 5 || isempty(nMC), nMC = cfg0.testing.nMC; end
if nargin < 6 || isempty(tag), tag = sprintf('T%d_B%d', Tvals(1), B); end
if nargin < 7 || isempty(saveOutput), saveOutput = true; end
if nargin < 8 || isempty(useParallel), useParallel = false; end
if nargin < 9 || isempty(cov_opts), cov_opts = empirical_covariance(cfg0); end
if nargin < 10 || isempty(designB), designB = B; end

Tvals = Tvals(:).';
B = double(B);
if ~isempty(lambda), lambda = double(lambda); end
% lambda may be [] (CV), a scalar, or an N-by-2 [seed, lambda_fit] lookup for
% the deterministic fixed-lambda replay. Resolved to one value per job below.
seed0 = double(seed0);
nMC = double(nMC);
tag = char(tag);
saveOutput = logical(saveOutput);
useParallel = logical(useParallel);
cov_opts = local_default_cov_opts(cov_opts);
covLabel = local_covariance_label(cov_opts);
% fixed_lambda: bind the supplied lambda (fingerprint, lambda_<tag> results folder).
cfg0 = apply_fixed_lambda(cfg0, lambda);
design_fingerprint = build_design_fingerprint(cfg0, Tvals, designB, nMC);
revision_id = cfg0.revision.id;

assert(B >= 1 && B == floor(B), 'B must be a positive integer.');
assert(nMC >= 1 && nMC == floor(nMC), 'nMC must be a positive integer.');
if isempty(lambda) && strcmp(cfg0.training.lambda_method, 'cv')
    assert(all(Tvals > cfg0.training.cv.min_train_dates), ...
        'Every T must exceed the simulation CV initial training window.');
end

testTypes = cfg0.testing.testTypes;
levels = [0.01, 0.05, 0.10];
regularizationLambdaPath = cfg0.diagnostics.regularization_lambda_path(:);

%% Build job grid and deterministic T-specific objects
nJobs = numel(Tvals) * B;
jobT = zeros(nJobs, 1);
jobRep = zeros(nJobs, 1);
jobSeed = zeros(nJobs, 1);

job = 0;
for iT = 1:numel(Tvals)
    for b = 1:B
        job = job + 1;
        jobT(job) = Tvals(iT);
        jobRep(job) = b;
        jobSeed(job) = seed0 + 1000000 * (iT - 1) + b - 1;
    end
end
jobLambda = resolve_job_lambda(lambda, jobSeed);

preByT = cell(numel(Tvals), 1);
for iT = 1:numel(Tvals)
    cfgT = cfg0;
    cfgT.sim.T = Tvals(iT);

    mon_grid = linspace(cfgT.training.estimator_domain(1), ...
        cfgT.training.estimator_domain(2), cfgT.training.n_m_grid).';
    option_m_grid = chj_option_moneyness_grid(cfgT);

    % The empirical kernel lengthscale is computed from the full option-level
    % training dataframe. In the balanced CHJ panel, each option moneyness
    % appears once on every date, so use its exact repeated-sample multiplicity.
    theta_m = median_pairwise_lengthscale_weighted( ...
        option_m_grid, repmat(cfgT.sim.T, numel(option_m_grid), 1));
    cfgT.kernel.endo.lengthscale = theta_m;
    cfgT.training.mon_grid = mon_grid;

    preT = struct();
    preT.cfg = cfgT;
    preT.mon_grid = mon_grid;
    preT.option_m_grid = option_m_grid;
    preT.theta_m = theta_m;
    preT.pre_base = build_chj_alpha_precompute(cfgT, mon_grid, []);
    preT.grids = build_region_grids(cfgT);
    preByT{iT} = preT;
end

preByJob = cell(nJobs, 1);
for j = 1:nJobs
    iT = find(Tvals == jobT(j), 1);
    preByJob{j} = preByT{iT};
end

fprintf('\nCHJ revision shape-rejection study\n');
fprintf('  T values: %s\n', mat2str(Tvals));
fprintf('  B: %d independent datasets per T\n', B);
if isempty(lambda) && strcmp(cfg0.training.lambda_method, 'eff_dim')
    fprintf('  lambda: effective dimension N_T(lambda) = %g * T^(1/3) (no CV)\n', ...
        cfg0.training.effdim_c);
elseif isempty(lambda)
    fprintf('  lambda: one-step-ahead CV on 31-point grid\n');
else
    fprintf('  lambda: fixed override %.6g\n', lambda);
end
fprintf('  CV initial training dates: %d\n', cfg0.training.cv.min_train_dates);
fprintf('  moneyness lengthscale rule: %s\n', cfg0.kernel.endo.lengthscale_rule);
fprintf('  VIX lengthscale rule: %s\n', cfg0.kernel.exo.lengthscale_rule);
fprintf('  estimator grid: %d moneyness x %d VIX points; VIX support: replication q05-q95\n', ...
    cfg0.training.n_m_grid, cfg0.training.n_exo_grid);
fprintf('  exogenous averaging: %s\n', cfg0.testing.exo.convention);
fprintf('  nMC: %d\n', nMC);
fprintf('  seed0: %d\n', seed0);
fprintf('  useParallel: %d\n', useParallel);
fprintf('  covariance: %s\n', covLabel);
fprintf('  inference lambda rule: %s\n', cfg0.testing.inference_lambda.rule);
fprintf('  per-rep regularization path: %d\n\n', ...
    cfg0.diagnostics.run_per_rep_regularization_path);

%% Run independent replication jobs
rawCells = cell(nJobs, 1);
regCells = cell(nJobs, 1);
recoveryCells = cell(nJobs, 1);
if useParallel
    pool = gcp('nocreate');
    if isempty(pool)
        parpool;
    end

    parfor j = 1:nJobs
        [rawCells{j}, regCells{j}, recoveryCells{j}] = local_one_replication( ...
            preByJob{j}, jobT(j), jobRep(j), jobSeed(j), jobLambda{j}, nMC, ...
            testTypes, levels, cov_opts, regularizationLambdaPath);
    end
else
    for j = 1:nJobs
        [rawCells{j}, regCells{j}, recoveryCells{j}] = local_one_replication( ...
            preByJob{j}, jobT(j), jobRep(j), jobSeed(j), jobLambda{j}, nMC, ...
            testTypes, levels, cov_opts, regularizationLambdaPath);

        if mod(j, max(1, floor(nJobs / 10))) == 0 || j == nJobs
            fprintf('  completed %d / %d jobs\n', j, nJobs);
        end
    end
end

raw = cat(1, rawCells{:});
regularization_bias_raw = local_cat_nonempty_tables(regCells);
recovery_raw = local_cat_nonempty_tables(recoveryCells);
summary = summarize_chj_rejection_results(raw, cfg0.training.cv.lambda_grid);

%% Save one block/checkpoint object when requested
if saveOutput
    outDir = fullfile(cfg0.results.path, 'rejection_studies', tag);
    if ~isfolder(outDir), mkdir(outDir); end

    rawFile = fullfile(outDir, sprintf('raw_%s.csv', tag));
    summaryFile = fullfile(outDir, sprintf('summary_%s.csv', tag));
    regFile = fullfile(outDir, sprintf('regularization_bias_%s.csv', tag));
    recoveryFile = fullfile(outDir, sprintf('recovery_%s.csv', tag));
    matFile = fullfile(outDir, sprintf('%s.mat', tag));

    % Every checkpoint write is atomic: the covariance rule, block size and
    % design fingerprint live inside the saved objects, not in the filename.
    local_atomic_writetable(raw, rawFile);
    local_atomic_writetable(summary, summaryFile);
    if ~isempty(regularization_bias_raw), local_atomic_writetable(regularization_bias_raw, regFile); end
    if ~isempty(recovery_raw), local_atomic_writetable(recovery_raw, recoveryFile); end
    matTmp = [matFile '.tmp.mat'];
    if isfile(matTmp), delete(matTmp); end
    run_metadata = simulation_run_metadata(cfg0.training.lambda_method);
    save(matTmp, 'summary', 'raw', 'regularization_bias_raw', 'recovery_raw', ...
        'regularizationLambdaPath', 'Tvals', 'B', 'lambda', 'seed0', ...
        'nMC', 'tag', 'cov_opts', 'covLabel', 'revision_id', 'design_fingerprint', ...
        'run_metadata');
    [okMat, msgMat] = movefile(matTmp, matFile, 'f');
    assert(okMat, 'Could not finalize %s: %s', matFile, msgMat);

    fprintf('\nWrote CHJ rejection-study outputs:\n');
    fprintf('  %s\n', rawFile);
    fprintf('  %s\n', summaryFile);
    if ~isempty(regularization_bias_raw), fprintf('  %s\n', regFile); end
    if ~isempty(recovery_raw), fprintf('  %s\n', recoveryFile); end
    fprintf('  %s\n', matFile);
end

end

function [rows, regRows, recoveryRows] = local_one_replication(preT, Tval, rep, seed, lambda, nMC, testTypes, levels, cov_opts, regularizationLambdaPath)
%% Simulate one CHJ dataset, select lambda, and evaluate all shape tests.

cfg = preT.cfg;
cfg.sim.T = Tval;
cfg.sim.seed = seed;
cfg.kernel.endo.lengthscale = preT.theta_m;
cfg.training.mon_grid = preT.mon_grid;

% Simulate the stochastic CHJ path once. All replication-specific tuning and
% option-payoff objects below are functions of this same path.
path = simulate_chj_numeric_path(cfg, seed);

% Match the empirical full-data kernel convention exactly: VIX is repeated on
% every option row for a date, so its pairwise-distance median uses J copies of
% each date-level state. The weighted calculation avoids materializing T*J rows.
theta_v = median_pairwise_lengthscale_weighted( ...
    path.VIX, repmat(cfg.sim.J, Tval, 1));
exo_grid = build_chj_exo_grid(cfg, path.VIX);

cfg.kernel.exo.lengthscale = theta_v;
cfg.training.exo_grid = exo_grid;
cfg.training.n_exo_grid = numel(exo_grid);

pre = preT.pre_base;
pre.fixed_exo_grid = exo_grid;
pre.ls_exo = theta_v;

alpha_mat = build_chj_alpha_numeric(cfg, preT.mon_grid, seed, pre, path);
Tobs = size(alpha_mat, 2);

[K_h, K_M, K_V, basis] = build_kernel_norm_matrix(cfg, preT.mon_grid, exo_grid);

alpha_bar = mean(alpha_mat, 2);
Q_alpha = (alpha_mat * alpha_mat.') / Tobs;
Q_alpha = 0.5 * (Q_alpha + Q_alpha.');

%% Shape-blind lambda selection
effdimSel = [];
fixedLambda = strcmp(cfg.training.lambda_method, 'fixed_lambda');
useQR = (isempty(lambda) && strcmp(cfg.training.lambda_method, 'eff_dim')) || fixedLambda;
if fixedLambda
    % Fixed-lambda study: the supplied lambda is used as is, for the fit
    % and for inference. No selection rule runs. The fit uses the same QR
    % least-squares solver as the eff_dim path (same estimator, no normal equations).
    assert(isscalar(lambda) && lambda == cfg.training.lambda, ...
        'fixed_lambda replication received a lambda different from the bound design lambda.');
    cv = [];
    payoff = build_payoff_factor(alpha_mat);
    lambda_fit = lambda;
    lambda_mode = "fixed_lambda";
    cv_mean_loss = NaN;
    cv_n_valid_dates = 0;
elseif useQR
    % Effective-dimension rule on the triangular payoff factor and K_h
    % (reduced 930-dimensional form; no Gram matrix is diagonalized). The same
    % factor gives the QR solve of the fit below. No CV is run.
    cv = [];
    payoff = build_payoff_factor(alpha_mat);
    effdimSel = local_select_effdim(payoff, K_h, Tobs, cfg.training.effdim_c, ...
        sprintf('CHJ T=%d rep=%d seed=%d', Tval, rep, seed));
    lambda_fit = effdimSel.lambda;
    lambda_mode = "eff_dim";
    cv_mean_loss = NaN;
    cv_n_valid_dates = 0;
elseif isempty(lambda)
    alphaPanel = struct('alpha', alpha_mat);
    cv = cross_validate_lambda(cfg, alphaPanel, K_h);
    lambda_fit = cv.lambda;
    lambda_mode = "cv";

    idxLambda = find(cv.lambda_grid == lambda_fit, 1);
    cv_mean_loss = cv.mean_loss(idxLambda);
    cv_n_valid_dates = cv.n_valid_dates(idxLambda);
else
    cv = [];
    lambda_fit = lambda;
    lambda_mode = "fixed";
    cv_mean_loss = NaN;
    cv_n_valid_dates = 0;
end

% Shape inference always tests the fitted object; there is no second
% inference-lambda selection.
lambda_inference = lambda_fit;
inference_rule = "lambda_fit";
inference_mean_loss_increase = 0;
inference_se_loss_increase = 0;
inference_hac_bandwidth = NaN;
inference_n_valid_dates = NaN;

if useQR
    % QR least squares for the same estimator. A holds the triangular factor
    % R_sys (R_sys' R_sys = Q_alpha + lambda K_h); every later solve with A
    % uses it (local_solve_symmetric), so the normal equations are not formed.
    [h_star, R_sys, solver_status] = solve_projection_qr(payoff, K_h, lambda_fit);
    A = struct('R_sys', R_sys);
else
    A = Q_alpha + lambda_fit * K_h;
    A = 0.5 * (A + A.');
    [h_star, solver_status] = local_solve_projection(A, alpha_bar);
end

model_state = struct();
model_state.h_star = h_star(:);
model_state.H_star = reshape(h_star(:), basis.n + 1, basis.p);
model_state.lambda = lambda_fit;
model_state.mon_grid = basis.mon_grid(:);
model_state.exo_grid = exo_grid(:);
model_state.ls_endo = basis.ls_endo;
model_state.ls_exo = theta_v;
model_state.reference_point = basis.reference_point;
model_state.K_h = K_h;
model_state.K_M = K_M;
model_state.K_V = K_V;
model_state.alpha_bar = alpha_bar;
model_state.Q_alpha = Q_alpha;
model_state.alpha = alpha_mat;
model_state.T = Tobs;
model_state.Nh = numel(h_star);
model_state.cfg = cfg;

%% Exact empirical equal-date state convention
exo_opts = struct();
exo_opts.mode = 'equal_date';
exo_opts.observations = path.VIX;
exo_opts.clip_to_grid = true;
exo = build_exo_representer(cfg, model_state, [], exo_opts);

%% Stack derivative loadings once, then run inference at lambda_inference
[~, L_all, blocks] = local_build_all_loadings( ...
    cfg, model_state, preT.grids, exo, testTypes);

% Shape inference uses the fitted solution directly; no second solve is allowed.
A_inf = A;
h_inf = h_star;
inference_solver_status = solver_status;

theta_all = L_all.' * h_inf;
[~, S_all] = local_omega_from_alpha( ...
    alpha_mat, h_inf, K_h, A_inf, L_all, lambda_inference);

cov_now = local_resolve_cov_bandwidth(cov_opts, Tobs);
cov_rule = local_covariance_label(cov_now);
[Omega_all, ~, cov_info] = build_score_covariance(S_all, (1:Tobs).', cov_now);

% Do not add four dense 930-dimensional solves to every final replication.
% The regularization path remains available through the separate population
% diagnostic, and never enters lambda selection or rejection decisions.
regRows = table();
if cfg.diagnostics.run_per_rep_regularization_path
    [~, S_fit] = local_omega_from_alpha(alpha_mat, h_star, K_h, A, L_all, lambda_fit);
    [Omega_fit, ~] = build_score_covariance(S_fit, (1:Tobs).', cov_now);
    regRows = local_regularization_path(Q_alpha, alpha_bar, K_h, L_all, blocks, ...
        Omega_fit, regularizationLambdaPath, lambda_fit, Tval, rep, seed, ...
        preT.theta_m, theta_v, exo, cfg, cov_now.bandwidth);
end

% Capture recovery directly from the already fitted T=2000 production state.
recoveryRows = table();
if Tval == cfg.diagnostics.recovery_T
    try
        recoveryRows = build_chj_sdf_recovery_curve(cfg, model_state, path, Tval, rep, seed, pre);
    catch err
        % Fixed small lambda only: the date-wise HJ normalization is undefined when a
        % date's grid mean is non-positive; record the replication as undefined (NaN)
        % instead of aborting. The shape test of this replication is unaffected.
        if ~(fixedLambda && strcmp(err.identifier, 'build_chj_sdf_recovery_curve:NonPositiveMean'))
            rethrow(err);
        end
        recoveryRows = local_undefined_recovery(cfg, model_state, Tval, rep, seed, err.message);
    end
end

outCells = cell(numel(blocks), 1);
convCache = struct('region', "", 'Omega_hat', [], 'W_mc', [], 'mc_meta', []);

for ib = 1:numel(blocks)
    blk = blocks(ib);
    idx = blk.idx;

    theta_hat = theta_all(idx);
    Omega_hat = Omega_all(idx, idx);
    Omega_hat = 0.5 * (Omega_hat + Omega_hat.');
    S_block = S_all(:, idx);

    stat = solve_wald_projection(theta_hat, Omega_hat, S_block, Tobs);

    reuseConvCalibration = strcmp(blk.testType, 'conc') ...
        && strcmp(convCache.region, string(blk.grid.name));

    if reuseConvCalibration
        errOmega = norm(Omega_hat - convCache.Omega_hat, 'fro');
        scaleOmega = max(1, norm(Omega_hat, 'fro'));
        assert(errOmega <= 1e-12 * scaleOmega, ...
            'CHJ convexity and concavity covariance blocks must agree within a region.');

        W_mc = convCache.W_mc;
        mc_meta = convCache.mc_meta;
        W_valid = W_mc(isfinite(W_mc));
        if isempty(W_valid)
            p_value = NaN;
        else
            p_value = (sum(W_valid >= stat.W_T) + 1) / (numel(W_valid) + 1);
        end
    else
        mc_opts = struct();
        mc_opts.parallel = false;
        mc_opts.rng_seed = seed + 1000 * ib + numel(theta_hat);

        [p_value, ~, W_mc, mc_meta] = build_mc_pvalue( ...
            stat.W_T, Omega_hat, S_block, Tobs, levels(2), nMC, mc_opts);

        if strcmp(blk.testType, 'conv')
            convCache.region = string(blk.grid.name);
            convCache.Omega_hat = Omega_hat;
            convCache.W_mc = W_mc;
            convCache.mc_meta = mc_meta;
        end
        W_valid = W_mc(isfinite(W_mc));
    end

    if isempty(W_valid)
        critical_1pct = NaN;
        critical_5pct = NaN;
        critical_10pct = NaN;
    else
        critical_1pct = quantile(W_valid, 1 - levels(1));
        critical_5pct = quantile(W_valid, 1 - levels(2));
        critical_10pct = quantile(W_valid, 1 - levels(3));
    end

    reject_1pct = double(stat.W_T > critical_1pct);
    reject_5pct = double(stat.W_T > critical_5pct);
    reject_10pct = double(stat.W_T > critical_10pct);
    if ~isfinite(p_value)
        reject_1pct = NaN;
        reject_5pct = NaN;
        reject_10pct = NaN;
    end

    solver_status_string = local_solver_status_string(inference_solver_status);
    fit_solver_status_string = local_solver_status_string(solver_status);
    covType = "";
    if isfield(cov_info, 'type') && ~isempty(cov_info.type)
        covType = string(cov_info.type);
    end
    covBandwidth = NaN;
    if isfield(cov_info, 'bandwidth') && ~isempty(cov_info.bandwidth)
        covBandwidth = cov_info.bandwidth;
    end

    outCells{ib} = table(Tval, rep, seed, Tobs, cfg.sim.horizon_days, ...
        lambda_inference, lambda_fit, lambda_inference, inference_rule, ...
        inference_mean_loss_increase, inference_se_loss_increase, ...
        inference_hac_bandwidth, inference_n_valid_dates, ...
        lambda_mode, cv_mean_loss, cv_n_valid_dates, ...
        preT.theta_m, theta_v, string(cfg.kernel.endo.lengthscale_rule), ...
        string(cfg.kernel.exo.lengthscale_rule), min(exo_grid), max(exo_grid), ...
        numel(exo_grid), string(cfg.training.exo_grid_type), ...
        string(exo.mode), exo.n_clipped, exo.frac_clipped, ...
        string(blk.grid.name), string(blk.testType), string(local_chj_role(blk.grid.name, blk.testType)), ...
        blk.grid.bounds(1), blk.grid.bounds(2), ...
        stat.W_T, p_value, critical_1pct, critical_5pct, critical_10pct, ...
        reject_1pct, reject_5pct, reject_10pct, ...
        min(theta_hat), max(theta_hat), mean(theta_hat < -cfg.testing.tau), ...
        NaN, solver_status_string, NaN, fit_solver_status_string, ...
        string(cov_rule), covType, covBandwidth, ...
        string(cov_now.bandwidth_rule), double(cov_now.winsor_margin), ...
        mc_meta.n_valid, mc_meta.n_failed, ...
        'VariableNames', {'Tval', 'rep', 'seed', 'Tobs', 'horizon_days', ...
        'lambda', 'lambda_fit', 'lambda_inference', 'inference_rule', ...
        'inference_mean_loss_increase', 'inference_se_loss_increase', ...
        'inference_hac_bandwidth', 'inference_n_valid_dates', ...
        'lambda_mode', 'cv_mean_loss', 'cv_n_valid_dates', ...
        'ls_endo', 'ls_exo', 'ls_endo_rule', 'ls_exo_rule', ...
        'exo_grid_min', 'exo_grid_max', 'n_exo_grid', 'exo_grid_type', ...
        'exo_mode', 'exo_n_clipped', 'exo_frac_clipped', ...
        'region', 'testType', 'role', 'region_left', 'region_right', ...
        'W_T', 'p_value', 'critical_1pct', 'critical_5pct', 'critical_10pct', ...
        'reject_1pct', 'reject_5pct', 'reject_10pct', ...
        'min_theta', 'max_theta', 'frac_negative', ...
        'condition_number', 'solver_status', 'fit_condition_number', ...
        'fit_solver_status', 'covariance_rule', ...
        'covariance_type', 'covariance_bandwidth', ...
        'covariance_bandwidth_rule', 'winsor_margin', 'mc_n_valid', 'mc_n_failed'});
end

rows = cat(1, outCells{:});
if ~isempty(effdimSel)
    rows = [rows, effdim_replication_diagnostics(effdimSel, height(rows))];
end
if fixedLambda
    rows.solve_relative_residual = repmat(local_normal_residual(Q_alpha, K_h, ...
        lambda_fit, h_star, alpha_bar), height(rows), 1);
    % 1-norm condition estimate of Q_alpha + lambda K_h from its QR factor (cheap, triangular).
    rows.solve_condition_estimate = repmat(1 / rcond(A.R_sys)^2, height(rows), 1);
end

end

function rows = local_regularization_path(Q_alpha, alpha_bar, K_h, L_all, blocks, Omega_all, lambdaPath, lambdaCV, Tval, rep, seed, thetaM, thetaV, exo, cfg, hacBandwidth)
%% Evaluate shape-vector movement over a fixed prespecified lambda path.

cells = cell(numel(lambdaPath) * numel(blocks), 1);
k = 0;
for il = 1:numel(lambdaPath)
    lam = lambdaPath(il);
    A_lam = 0.5 * ((Q_alpha + lam * K_h) + (Q_alpha + lam * K_h).');
    h_lam = local_solve_symmetric(A_lam, alpha_bar);
    theta_all = L_all.' * h_lam;

    for ib = 1:numel(blocks)
        blk = blocks(ib);
        idx = blk.idx;
        theta = theta_all(idx);
        Omega = 0.5 * (Omega_all(idx, idx) + Omega_all(idx, idx).');
        stat = solve_wald_projection(theta, Omega, [], 1);

        mahal = NaN;
        if isfinite(stat.W_T)
            mahal = sqrt(max(stat.W_T, 0));
        end

        k = k + 1;
        cells{k} = table(Tval, rep, seed, lambdaCV, lam, thetaM, thetaV, ...
            string(blk.grid.name), string(blk.testType), ...
            string(local_chj_role(blk.grid.name, blk.testType)), numel(theta), ...
            min(theta), max(theta), mean(theta < -cfg.testing.tau), ...
            norm(min(theta, 0)), mahal, exo.n_clipped, exo.frac_clipped, ...
            hacBandwidth, "fixed_cv_hac", ...
            'VariableNames', {'Tval','rep','seed','lambda_cv','lambda', ...
            'ls_endo','ls_exo','region','testType','role','n_grid', ...
            'min_theta','max_theta','frac_negative','euclidean_cone_distance', ...
            'mahalanobis_cone_distance','exo_n_clipped','exo_frac_clipped', ...
            'hac_bandwidth','distance_metric'});
    end
end
rows = cat(1, cells{:});

end

function [theta_all, L_all, blocks] = local_build_all_loadings(cfg, model_state, grids, exo, testTypes)
%% Stack all CHJ derivative loadings in region/test order.

nBlocks = numel(grids) * numel(testTypes);
Lcells = cell(nBlocks, 1);
thetaCells = cell(nBlocks, 1);
blocks = repmat(struct('grid', [], 'testType', '', 'idx', []), nBlocks, 1);

offset = 0;
row = 0;
for ig = 1:numel(grids)
    grid = grids(ig);
    for it = 1:numel(testTypes)
        testType = testTypes{it};
        [theta_hat, L_Z] = build_theta_hat(cfg, model_state, grid, testType, exo);

        row = row + 1;
        q = numel(theta_hat);
        blocks(row).grid = grid;
        blocks(row).testType = testType;
        blocks(row).idx = offset + (1:q);
        offset = offset + q;

        thetaCells{row} = theta_hat(:);
        Lcells{row} = L_Z;
    end
end

theta_all = cat(1, thetaCells{:});
L_all = cat(2, Lcells{:});

end

function [Omega_hat, S] = local_omega_from_alpha(alpha_mat, h_star, K_h, A, L_all, lambda)
%% Build the finite-grid influence-score matrix for the simulated alpha panel.

U_all = local_solve_symmetric(A, L_all);
eta = alpha_mat.' * h_star;
alphaU = alpha_mat.' * U_all;
h_u = h_star.' * K_h * U_all;

S = bsxfun(@times, 1 - eta, alphaU) ...
    - ones(size(alpha_mat, 2), 1) * (lambda * h_u);
Omega_hat = (S.' * S) / size(S, 1);
Omega_hat = 0.5 * (Omega_hat + Omega_hat.');

end

function role = local_chj_role(region, testType)
%% Label each retained CHJ region/test cell by its known DGP role.

region = char(region);
testType = char(testType);

switch testType
    case 'conv'
        role = 'true_null_size';
    case 'conc'
        role = 'false_null_power';
    case 'mono'
        switch region
            case 'left'
                role = 'true_null_size';
            case 'center'
                role = 'transition_diagnostic';
            case 'right'
                role = 'false_null_power';
            otherwise
                role = 'diagnostic';
        end
    otherwise
        role = 'diagnostic';
end

end

function [h_star, solver_status] = local_solve_projection(A, b)
%% Solve the regularized projection system and record the solver path.

solver_status = struct('flag', 0, 'method', 'chol');
[L, p] = chol(A, 'lower');
if p == 0
    h_star = L' \ (L \ b);
    return
end

solver_status.flag = p;
solver_status.method = 'backslash';
h_star = A \ b;

end

function X = local_solve_symmetric(A, B)
%% Solve a symmetric linear system without changing the statistical object.
% A is either the system matrix (Cholesky path) or struct('R_sys', R) with
% R' R = A from solve_projection_qr (two triangular solves).

if isstruct(A)
    X = A.R_sys \ (A.R_sys.' \ B);
    return
end
A = 0.5 * (A + A.');
[L, p] = chol(A, 'lower');
if p == 0
    X = L' \ (L \ B);
else
    X = A \ B;
end

end

function status = local_solver_status_string(solver_status)
%% Return a compact linear-solver label.

if isfield(solver_status, 'method')
    status = string(solver_status.method);
elseif isfield(solver_status, 'status')
    status = string(solver_status.status);
elseif isfield(solver_status, 'reason')
    status = string(solver_status.reason);
else
    status = "ok";
end

end

function out = local_cat_nonempty_tables(cells)
%% Concatenate optional per-replication tables without 0-by-0 table conflicts.

keep = ~cellfun(@isempty, cells);
if any(keep)
    out = cat(1, cells{keep});
else
    out = table();
end

end

function cov_opts = empirical_covariance(cfg)
%% Return the common empirical Bartlett HAC rule.

cov_opts = struct();
cov_opts.type = 'hac_bartlett';
cov_opts.bandwidth = 0;
cov_opts.bandwidth_rule = cfg.testing.hac_bandwidth_rule;
cov_opts.winsor_margin = 0;
cov_opts.recenter = false;

end

function cov_opts = local_default_cov_opts(cov_opts)
%% Fill covariance options used by the CHJ runner.

if nargin < 1 || isempty(cov_opts), cov_opts = struct(); end
if ~isfield(cov_opts, 'type') || isempty(cov_opts.type), cov_opts.type = 'hac_bartlett'; end
if ~isfield(cov_opts, 'bandwidth') || isempty(cov_opts.bandwidth), cov_opts.bandwidth = 0; end
if ~isfield(cov_opts, 'bandwidth_rule') || isempty(cov_opts.bandwidth_rule), cov_opts.bandwidth_rule = 'nw_4_t100_2over9'; end
if ~isfield(cov_opts, 'winsor_margin') || isempty(cov_opts.winsor_margin), cov_opts.winsor_margin = 0; end
if ~isfield(cov_opts, 'recenter') || isempty(cov_opts.recenter), cov_opts.recenter = false; end

cov_opts.type = char(lower(string(cov_opts.type)));
cov_opts.bandwidth_rule = char(lower(string(cov_opts.bandwidth_rule)));
cov_opts.bandwidth = double(cov_opts.bandwidth);
cov_opts.winsor_margin = double(cov_opts.winsor_margin);

end

function cov_opts = local_resolve_cov_bandwidth(cov_opts, T)
%% Resolve the common Bartlett lag rule for one sample size.

cov_opts = local_default_cov_opts(cov_opts);
rule = char(lower(string(cov_opts.bandwidth_rule)));
if isempty(rule), return; end

switch rule
    case 'nw_4_t100_2over9'
        cov_opts.bandwidth = bartlett_hac_bandwidth(T);
    otherwise
        error('run_chj_simulated_shape_rejection_study:BadBandwidthRule', ...
            'CHJ revision requires the empirical HAC bandwidth rule %s.', ...
            'nw_4_t100_2over9');
end

end

function label = local_covariance_label(cov_opts)
%% Build a compact covariance label for output metadata.

cov_opts = local_default_cov_opts(cov_opts);
covType = char(lower(string(cov_opts.type)));
rule = char(lower(string(cov_opts.bandwidth_rule)));
bandwidth = double(cov_opts.bandwidth);

if strcmp(covType, 'iid_lag0') && bandwidth == 0 && isempty(rule)
    label = 'iid_lag0';
elseif ~isempty(rule)
    label = sprintf('%s_%s', covType, rule);
else
    label = sprintf('%s_L%d', covType, bandwidth);
end
label = regexprep(label, '[^A-Za-z0-9_]', '_');

end

function local_atomic_writetable(tbl, file)
%% Write one CSV atomically: fill a temporary file, then rename into place.
%
% An interrupted write therefore leaves the previous checkpoint intact rather
% than a truncated file.

tmp = [file '.tmp.csv'];
if isfile(tmp), delete(tmp); end
writetable(tbl, tmp);
[ok, msg] = movefile(tmp, file, 'f');
assert(ok, 'Could not finalize %s: %s', file, msg);

end

function sel = local_select_effdim(payoff, K_h, Tobs, c, context)
%% Effective-dimension selection; an unattainable target stops the run.
% The selector's error is rethrown with the replication identity (same
% identifier) so the infeasible seed is reported. There is no fallback.

try
    sel = select_lambda_effective_dimension(payoff, K_h, Tobs, c);
catch ME
    if strcmp(ME.identifier, 'select_lambda_effective_dimension:Unattainable')
        error('select_lambda_effective_dimension:Unattainable', ...
            '%s: %s The effective-dimension target exceeds this sample''s payoff rank; the run stops (no fallback).', ...
            context, ME.message);
    end
    rethrow(ME);
end

end

function r = local_normal_residual(Q_alpha, K_h, lambda, h, b)
%% Relative backward residual of (Q_alpha + lambda K_h) h = alpha_bar (diagnostic only).
A = Q_alpha + lambda * K_h;
r = norm(A * h - b) / max(norm(A, 'fro') * norm(h) + norm(b), realmin);
end

function rows = local_undefined_recovery(cfg, model_state, Tval, rep, seed, msg)
%% NaN recovery rows (same schema as build_chj_sdf_recovery_curve) for an undefined HJ normalization.
m = build_global_shape_grid(cfg); m = m(:); n = numel(m);
tok = regexp(msg, 'min ([-+0-9.eE]+)\)', 'tokens', 'once');
hjMin = NaN; if ~isempty(tok), hjMin = str2double(tok{1}); end
rows = table(repmat(Tval,n,1), repmat(rep,n,1), repmat(seed,n,1), ...
    repmat(model_state.lambda,n,1), m, nan(n,1), nan(n,1), nan(n,1), ...
    nan(n,1), repmat(hjMin,n,1), repmat("hj_undefined",n,1), ...
    'VariableNames', {'Tval','rep','seed','lambda','m', ...
    'estimated_sdf','true_sdf','difference','exo_frac_outside_grid', ...
    'hj_min_date_grid_mean','recovery_measure'});
end
