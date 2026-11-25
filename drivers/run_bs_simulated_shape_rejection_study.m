function [summary, raw, regularization_bias_raw, recovery_raw] = run_bs_simulated_shape_rejection_study(Tvals, B, lambda, seed0, nMC, tag, saveOutput, useParallel, cov_opts, horizon, experiment, designB, lambdaMethod)
%% Run independent Black--Scholes learning and shape-test replications.
%
% Each replication generates a fresh BS option-payoff panel, selects lambda
% by the configured rule (lambdaMethod: 'cv', the default one-step-ahead CV,
% or 'eff_dim', the effective-dimension rule) unless a fixed value is
% explicitly supplied, estimates
% h, and runs the finite-grid shape tests. Seed-invariant kernel, option-price,
% and derivative-loading objects are built once and reused across replications.

setpath();

if nargin < 10 || isempty(horizon), horizon = 'monthly'; end
if nargin < 11 || isempty(experiment), experiment = 'bs_increasing_t'; end
if nargin < 13 || isempty(lambdaMethod), lambdaMethod = 'cv'; end
cfg0 = config_bs_simulation_design(horizon, experiment, lambdaMethod);
revision_id = cfg0.revision.id;

if nargin < 1 || isempty(Tvals), Tvals = cfg0.sim.empirical_T; end
if nargin < 2 || isempty(B), B = 100; end
if nargin < 3, lambda = cfg0.training.lambda; end
if nargin < 4 || isempty(seed0), seed0 = cfg0.sim.seed0; end
if nargin < 5 || isempty(nMC), nMC = cfg0.testing.nMC; end
if nargin < 6 || isempty(tag), tag = 'bs_shape_rejection'; end
if nargin < 7 || isempty(saveOutput), saveOutput = true; end
if nargin < 8 || isempty(useParallel), useParallel = false; end
if nargin < 9 || isempty(cov_opts), cov_opts = struct(); end
if nargin < 12 || isempty(designB), designB = B; end

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

assert(B >= 1 && B == floor(B), 'B must be a positive integer.');
assert(nMC >= 1 && nMC == floor(nMC), 'nMC must be a positive integer.');
if isempty(lambda) && strcmp(cfg0.training.lambda_method, 'cv')
    assert(all(Tvals > cfg0.training.cv.min_train_dates), ...
        'Every T must exceed cfg.training.cv.min_train_dates.');
end

fixed = local_precompute_design(cfg0);
levels = [0.01, 0.05, 0.10];
regularizationLambdaPath = cfg0.diagnostics.regularization_lambda_path(:);

%% Build independent replication jobs
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

rawCells = cell(nJobs, 1);
regCells = cell(nJobs, 1);
recoveryCells = cell(nJobs, 1);

fprintf('\nBS simulated shape-rejection study\n');
fprintf('  horizon: %s\n', cfg0.sim.horizon);
fprintf('  maturity years: %.8g\n', cfg0.bs.dt);
fprintf('  kernel lengthscale: %.8g (%s rule)\n', fixed.basis.ls_endo, ...
    cfg0.kernel.endo.lengthscale_rule);
fprintf('  T values: %s\n', mat2str(Tvals));
fprintf('  B: %d independent datasets per T\n', B);
jobLambda = resolve_job_lambda(lambda, jobSeed);
if isempty(lambda) && strcmp(cfg0.training.lambda_method, 'eff_dim')
    fprintf('  lambda: effective dimension N_T(lambda) = %g * T^(1/3) (no CV)\n', ...
        cfg0.training.effdim_c);
elseif isempty(lambda)
    fprintf('  lambda: one-step-ahead CV on %d-point grid [%.3g, %.3g]\n', ...
        numel(cfg0.training.cv.lambda_grid), min(cfg0.training.cv.lambda_grid), ...
        max(cfg0.training.cv.lambda_grid));
elseif isscalar(lambda)
    fprintf('  lambda: %.6g (fixed, no CV)\n', lambda);
else
    fprintf('  lambda: saved per-replication lambda_fit for %d seeds (fixed, no CV)\n', ...
        size(lambda,1));
end
fprintf('  nMC: %d\n', nMC);
fprintf('  seed0: %d\n', seed0);
fprintf('  useParallel: %d\n', useParallel);
fprintf('  covariance: %s\n', covLabel);
fprintf('  CV lambda scale: absolute\n');
fprintf('  inference lambda rule: %s\n\n', cfg0.testing.inference_lambda.rule);

%% Run independent learning/testing jobs
if useParallel
    pool = gcp('nocreate');
    if isempty(pool)
        parpool;
    end

    parfor job = 1:nJobs
        [rawCells{job}, regCells{job}, recoveryCells{job}] = local_one_replication( ...
            cfg0, fixed, jobT(job), jobRep(job), jobSeed(job), jobLambda{job}, nMC, ...
            levels, cov_opts, regularizationLambdaPath);
    end
else
    for job = 1:nJobs
        [rawCells{job}, regCells{job}, recoveryCells{job}] = local_one_replication( ...
            cfg0, fixed, jobT(job), jobRep(job), jobSeed(job), jobLambda{job}, nMC, ...
            levels, cov_opts, regularizationLambdaPath);

        if mod(job, max(1, floor(nJobs / 10))) == 0 || job == nJobs
            fprintf('  completed %d / %d jobs\n', job, nJobs);
        end
    end
end

raw = cat(1, rawCells{:});
regularization_bias_raw = local_cat_nonempty_tables(regCells);
recovery_raw = local_cat_nonempty_tables(recoveryCells);
summary = summarize_bs_rejection_results(raw, cfg0.training.cv.lambda_grid);

%% Save replication-level and aggregate outputs
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
        'nMC', 'tag', 'cov_opts', 'covLabel', 'horizon', 'experiment', ...
        'revision_id', 'design_fingerprint', 'run_metadata');
    [okMat, msgMat] = movefile(matTmp, matFile, 'f');
    assert(okMat, 'Could not finalize %s: %s', matFile, msgMat);

    fprintf('\nWrote BS rejection-study outputs:\n');
    fprintf('  %s\n', rawFile);
    fprintf('  %s\n', summaryFile);
    if ~isempty(regularization_bias_raw), fprintf('  %s\n', regFile); end
    if ~isempty(recovery_raw), fprintf('  %s\n', recoveryFile); end
    fprintf('  %s\n', matFile);
end

end

function fixed = local_precompute_design(cfg)
%% Build seed-invariant BS option, kernel, grid, and derivative objects once.

mon_grid = linspace(cfg.training.estimator_domain(1), ...
    cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
cfg.training.mon_grid = mon_grid;

pre = build_bs_alpha_precompute(cfg, mon_grid);
cfg.kernel.endo.lengthscale = pre.ell_m;
[K_h, K_M, K_V, basis] = build_kernel_norm_matrix(cfg, mon_grid, []);
grids = local_build_bs_grids(cfg);

template = struct();
template.h_star = zeros((basis.n + 1) * basis.p, 1);
template.mon_grid = basis.mon_grid;
template.exo_grid = basis.exo_grid;
template.ls_endo = basis.ls_endo;
template.ls_exo = basis.ls_exo;
template.reference_point = basis.reference_point;

exo = build_exo_representer(cfg, template, struct());
[~, L_all, blocks] = local_build_all_loadings(cfg, template, grids, exo, cfg.testing.testTypes);

fixed = struct();
fixed.mon_grid = mon_grid;
fixed.pre = pre;
fixed.K_h = K_h;
fixed.K_M = K_M;
fixed.K_V = K_V;
fixed.basis = basis;
fixed.L_all = L_all;
fixed.blocks = blocks;

end

function [rows, regRows, recoveryRows] = local_one_replication(cfg0, fixed, Tval, rep, seed, lambda, nMC, levels, cov_opts, regularizationLambdaPath)
%% Simulate one BS dataset, learn h, and evaluate every retained shape test.

cfg = cfg0;
cfg.sim.T = Tval;
cfg.sim.seed = seed;
cfg.training.lambda = lambda;
cfg.training.use_cv = isempty(lambda) && strcmp(cfg0.training.lambda_method, 'cv');
cfg.training.mon_grid = fixed.mon_grid;
cfg.training.n_m_grid = numel(fixed.mon_grid);
cfg.training.exo_grid = [];
cfg.testing.nMC = nMC;

alpha_mat = build_bs_alpha_numeric(cfg, fixed.mon_grid, seed, fixed.pre);
Tobs = size(alpha_mat, 2);
alpha_bar = mean(alpha_mat, 2);
Q_alpha = (alpha_mat * alpha_mat.') / Tobs;
Q_alpha = 0.5 * (Q_alpha + Q_alpha.');

effdimSel = [];
fixedLambda = strcmp(cfg0.training.lambda_method, 'fixed_lambda');
useQR = (isempty(lambda) && strcmp(cfg0.training.lambda_method, 'eff_dim')) || fixedLambda;
if fixedLambda
    % Fixed-lambda study: the supplied lambda is used as is, for the fit
    % and for inference. No selection rule runs. The fit uses the same QR
    % least-squares solver as the eff_dim path (same estimator, no normal equations).
    assert(isscalar(lambda) && lambda == cfg0.training.lambda, ...
        'fixed_lambda replication received a lambda different from the bound design lambda.');
    cv = [];
    payoff = build_payoff_factor(alpha_mat);
    lambda_fit = lambda;
    lambda_multiplier = NaN;
    lambda_scale = NaN;
    lambda_scale_mode = "fixed_absolute";
    cv_mean_loss = NaN;
    cv_n_valid_dates = NaN;
    cv_upper_rel_excess = NaN;
    cv_upper_step_rel_improvement = NaN;
    cv_upper_decade_rel_range = NaN;
    lambda_mode = "fixed_lambda";
elseif useQR
    % Effective-dimension rule on the triangular payoff factor and K_h; the
    % same factor gives the QR solve of the fit below. No CV is run.
    cv = [];
    payoff = build_payoff_factor(alpha_mat);
    effdimSel = local_select_effdim(payoff, fixed.K_h, Tobs, cfg0.training.effdim_c, ...
        sprintf('BS %s %s T=%d rep=%d seed=%d', cfg0.sim.experiment, cfg0.sim.horizon, Tval, rep, seed));
    lambda_fit = effdimSel.lambda;
    lambda_multiplier = NaN;
    lambda_scale = NaN;
    lambda_scale_mode = "eff_dim";
    cv_mean_loss = NaN;
    cv_n_valid_dates = NaN;
    cv_upper_rel_excess = NaN;
    cv_upper_step_rel_improvement = NaN;
    cv_upper_decade_rel_range = NaN;
    lambda_mode = "eff_dim";
elseif isempty(lambda)
    alphaPanel = struct('alpha', alpha_mat);
    cv = cross_validate_lambda(cfg, alphaPanel, fixed.K_h);
    lambda_fit = cv.lambda;
    lambda_multiplier = cv.lambda_multiplier;
    lambda_scale = cv.lambda_scale;
    lambda_scale_mode = string(cv.lambda_scale_mode);
    cv_mean_loss = cv.mean_loss(cv.selected_index);
    cv_n_valid_dates = cv.n_valid_dates(cv.selected_index);
    [cv_upper_rel_excess, cv_upper_step_rel_improvement, ...
        cv_upper_decade_rel_range] = local_cv_flatness(cv);
    lambda_mode = "cv";
else
    cv = [];
    lambda_fit = lambda;
    lambda_multiplier = NaN;
    lambda_scale = NaN;
    lambda_scale_mode = "fixed_absolute";
    cv_mean_loss = NaN;
    cv_n_valid_dates = NaN;
    cv_upper_rel_excess = NaN;
    cv_upper_step_rel_improvement = NaN;
    cv_upper_decade_rel_range = NaN;
    lambda_mode = "fixed";
end

% Shape inference always tests the fitted object; there is no second
% inference-lambda selection.
lambda_inference = lambda_fit;
inference_rule = "lambda_fit";
inference_mean_loss_increase = 0;
inference_se_loss_increase = 0;
inference_hac_bandwidth = NaN;
inference_n_valid_dates = NaN;

cfg.training.lambda = lambda_fit;
R_sys = [];
if useQR
    % QR least squares for the same estimator; R_sys' R_sys = Q_alpha + lambda K_h.
    [h_star, R_sys, solver_status] = solve_projection_qr(payoff, fixed.K_h, lambda_fit);
    conditionNumber = cond(R_sys)^2;
else
    A = Q_alpha + lambda_fit * fixed.K_h;
    A = 0.5 * (A + A.');
    [h_star, solver_status] = local_solve_projection(A, alpha_bar);
    conditionNumber = cond(A);
end

model_state = struct();
model_state.alpha = alpha_mat;
model_state.h_star = h_star(:);
model_state.H_star = reshape(h_star(:), fixed.basis.n + 1, fixed.basis.p);
model_state.lambda = lambda_fit;
model_state.cv = cv;
model_state.mon_grid = fixed.basis.mon_grid;
model_state.exo_grid = fixed.basis.exo_grid;
model_state.ls_endo = fixed.basis.ls_endo;
model_state.ls_exo = fixed.basis.ls_exo;
model_state.reference_point = fixed.basis.reference_point;
model_state.K_h = fixed.K_h;
model_state.K_M = fixed.K_M;
model_state.K_V = fixed.K_V;
model_state.alpha_bar = alpha_bar;
model_state.Q_alpha = Q_alpha;
model_state.T = Tobs;
model_state.Nh = numel(h_star);
model_state.condition_number = conditionNumber;
model_state.solver_status = solver_status;
model_state.solve_factor = R_sys;      % empty on the CV/fixed path (Cholesky)
model_state.cfg = cfg;

% Shape inference uses the fitted state directly; no second solve is allowed.
inference_state = model_state;

theta_all = fixed.L_all.' * inference_state.h_star;
[~, S_all] = build_Omega_hat_ls(inference_state, fixed.L_all);
cov_now = local_resolve_cov_bandwidth(cov_opts, Tobs);
cov_rule = local_covariance_label(cov_now);
[Omega_all, ~, cov_info] = build_score_covariance(S_all, (1:Tobs).', cov_now);

% Per-replication regularization sensitivity is disabled in production.
regRows = table();
if cfg.diagnostics.run_per_rep_regularization_path
    [~, S_fit] = build_Omega_hat_ls(model_state, fixed.L_all);
    [Omega_fit, ~] = build_score_covariance(S_fit, (1:Tobs).', cov_now);
    regRows = local_regularization_path(Q_alpha, alpha_bar, fixed.K_h, fixed.L_all, ...
        fixed.blocks, Omega_fit, regularizationLambdaPath, lambda_fit, Tval, ...
        rep, seed, cfg, cov_now.bandwidth);
end

% Recovery is captured during the T=2000 monthly production fit itself. This
% avoids regenerating/refitting the same 1000 datasets in a second pass.
recoveryRows = table();
if strcmpi(cfg.sim.horizon, 'monthly') && Tval == cfg.diagnostics.recovery_T
    recoveryRows = build_bs_sdf_recovery_curve(cfg, model_state, Tval, rep, seed, fixed.pre);
end

outCells = cell(numel(fixed.blocks), 1);

% Convexity and concavity use opposite second-derivative loadings. Their
% covariance matrices are therefore identical within a region, so the same
% Monte Carlo null calibration can be reused exactly.
convCache = struct('region', "", 'Omega_hat', [], 'W_mc', [], 'mc_meta', []);

for ib = 1:numel(fixed.blocks)
    blk = fixed.blocks(ib);
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
            'Convexity and concavity covariance blocks must agree within a region.');

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

    solver_status_string = local_solver_status_string(inference_state.solver_status);
    fit_solver_status_string = local_solver_status_string(solver_status);

    covType = '';
    if isfield(cov_info, 'type') && ~isempty(cov_info.type)
        covType = cov_info.type;
    end

    covBandwidth = NaN;
    if isfield(cov_info, 'bandwidth') && ~isempty(cov_info.bandwidth)
        covBandwidth = cov_info.bandwidth;
    end

    outCells{ib} = table(string(cfg.sim.horizon), Tval, rep, seed, Tobs, ...
        lambda_inference, lambda_fit, lambda_inference, inference_rule, ...
        inference_mean_loss_increase, inference_se_loss_increase, ...
        inference_hac_bandwidth, inference_n_valid_dates, ...
        lambda_multiplier, lambda_scale, lambda_scale_mode, ...
        lambda_mode, cv_mean_loss, cv_n_valid_dates, ...
        cv_upper_rel_excess, cv_upper_step_rel_improvement, ...
        cv_upper_decade_rel_range, ...
        string(blk.grid.name), blk.grid.bounds(1), blk.grid.bounds(2), ...
        string(blk.testType), stat.W_T, p_value, ...
        critical_1pct, critical_5pct, critical_10pct, ...
        reject_1pct, reject_5pct, reject_10pct, ...
        min(theta_hat), max(theta_hat), mean(theta_hat < -cfg.testing.tau), ...
        inference_state.condition_number, solver_status_string, ...
        model_state.condition_number, fit_solver_status_string, ...
        string(cov_rule), string(covType), covBandwidth, ...
        string(cov_now.bandwidth_rule), double(cov_now.winsor_margin), ...
        mc_meta.n_valid, mc_meta.n_failed, ...
        'VariableNames', {'horizon', 'Tval', 'rep', 'seed', 'Tobs', ...
        'lambda', 'lambda_fit', 'lambda_inference', 'inference_rule', ...
        'inference_mean_loss_increase', 'inference_se_loss_increase', ...
        'inference_hac_bandwidth', 'inference_n_valid_dates', ...
        'lambda_multiplier', 'lambda_scale', 'lambda_scale_mode', ...
        'lambda_mode', 'cv_mean_loss', 'cv_n_valid_dates', ...
        'cv_upper_rel_excess', 'cv_upper_step_rel_improvement', ...
        'cv_upper_decade_rel_range', ...
        'region', 'region_left', 'region_right', 'testType', ...
        'W_T', 'p_value', 'critical_1pct', 'critical_5pct', 'critical_10pct', ...
        'reject_1pct', 'reject_5pct', 'reject_10pct', ...
        'min_theta', 'max_theta', 'frac_negative', ...
        'condition_number', 'solver_status', 'fit_condition_number', ...
        'fit_solver_status', ...
        'covariance_rule', 'covariance_type', 'covariance_bandwidth', ...
        'covariance_bandwidth_rule', 'winsor_margin', ...
        'mc_n_valid', 'mc_n_failed'});
end

rows = cat(1, outCells{:});
if ~isempty(effdimSel)
    rows = [rows, effdim_replication_diagnostics(effdimSel, height(rows))];
end
if fixedLambda
    rows.solve_relative_residual = repmat(local_normal_residual(Q_alpha, fixed.K_h, ...
        lambda_fit, h_star, alpha_bar), height(rows), 1);
end

end

function grids = local_build_bs_grids(cfg)
%% Build the common midpoint shape-test grid without writing generated files.

specs = cfg.testing.regionSpecs;
mAll = build_global_shape_grid(cfg);

grids = repmat(struct('name', '', 'label', '', 'bounds', [], 'm', []), numel(specs), 1);
for j = 1:numel(specs)
    b = specs(j).bounds;
    if j < numel(specs)
        idx = mAll >= b(1) & mAll < b(2);
    else
        idx = mAll >= b(1) & mAll <= b(2);
    end
    assert(any(idx), 'BS shape-test region %s has no grid points.', specs(j).name);
    grids(j).name = specs(j).name;
    grids(j).label = specs(j).label;
    grids(j).bounds = b;
    grids(j).m = mAll(idx);
end

end

function [theta_all, L_all, blocks] = local_build_all_loadings(cfg, model_state, grids, exo, testTypes)
%% Stack all fixed BS derivative loadings in the retained test order.

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

function rows = local_regularization_path(Q_alpha, alpha_bar, K_h, L_all, blocks, Omega_all, lambdaPath, lambdaCV, Tval, rep, seed, cfg, hacBandwidth)
%% Evaluate the prespecified fixed-HAC regularization path on one BS dataset.

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
        if isfinite(stat.W_T), mahal = sqrt(max(stat.W_T, 0)); end

        k = k + 1;
        cells{k} = table(Tval, rep, seed, lambdaCV, lam, ...
            string(blk.grid.name), string(blk.testType), ...
            string(local_bs_structural_role(blk.testType)), numel(theta), ...
            min(theta), max(theta), mean(theta < -cfg.testing.tau), ...
            norm(min(theta, 0)), mahal, hacBandwidth, "fixed_cv_hac", ...
            'VariableNames', {'Tval','rep','seed','lambda_cv','lambda', ...
            'region','testType','role','n_grid','min_theta','max_theta', ...
            'frac_negative','euclidean_cone_distance', ...
            'mahalanobis_cone_distance','hac_bandwidth','distance_metric'});
    end
end
rows = cat(1, cells{:});

end

function role = local_bs_structural_role(testType)
%% Label only the known structural BS restriction, not finite-lambda test size.

switch char(testType)
    case {'mono','conv'}
        role = 'structural_null_holds';
    case 'conc'
        role = 'structural_null_fails';
    otherwise
        role = 'diagnostic';
end

end

function [upperRelExcess, upperStepRelImprovement, upperDecadeRelRange] = local_cv_flatness(cv)
%% Summarize whether the CV loss is still changing near the top of the grid.

loss = cv.mean_loss(:);
multiplier = cv.lambda_multiplier_grid(:);
good = isfinite(loss);
if ~any(good)
    upperRelExcess = NaN;
    upperStepRelImprovement = NaN;
    upperDecadeRelRange = NaN;
    return
end

minLoss = min(loss(good));
den = max(abs(minLoss), eps);
upperRelExcess = (loss(end) - minLoss) / den;

if numel(loss) >= 2 && isfinite(loss(end-1)) && isfinite(loss(end))
    upperStepRelImprovement = (loss(end-1) - loss(end)) / den;
else
    upperStepRelImprovement = NaN;
end

top = good & multiplier >= max(multiplier) / 10;
if any(top)
    upperDecadeRelRange = (max(loss(top)) - min(loss(top))) / den;
else
    upperDecadeRelRange = NaN;
end
end

function X = local_solve_symmetric(A, B)
%% Solve a symmetric system without changing the statistical target.

A = 0.5 * (A + A.');
[L, p] = chol(A, 'lower');
if p == 0
    X = L' \ (L \ B);
else
    X = A \ B;
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

function [h_star, solver_status] = local_solve_projection(A, b)
%% Solve the regularized projection system and record the solver path.

solver_status = struct();
solver_status.flag = 0;
solver_status.method = 'chol';

[L, p] = chol(A, 'lower');
if p == 0
    h_star = L' \ (L \ b);
    return
end

solver_status.flag = p;
solver_status.method = 'backslash';
h_star = A \ b;

end

function status = local_solver_status_string(solver_status)
%% Return a compact solver-status label for replication output.

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

function cov_opts = local_default_cov_opts(cov_opts)
%% Fill covariance options used by the generic BS runner.

if nargin < 1 || isempty(cov_opts)
    cov_opts = struct();
end
if ~isfield(cov_opts, 'type') || isempty(cov_opts.type), cov_opts.type = 'hac_bartlett'; end
if ~isfield(cov_opts, 'bandwidth') || isempty(cov_opts.bandwidth)
    cov_opts.bandwidth = 0;
end
if ~isfield(cov_opts, 'bandwidth_rule') || isempty(cov_opts.bandwidth_rule)
    cov_opts.bandwidth_rule = 'nw_4_t100_2over9';
end
if ~isfield(cov_opts, 'winsor_margin') || isempty(cov_opts.winsor_margin)
    cov_opts.winsor_margin = 0;
end
if ~isfield(cov_opts, 'recenter') || isempty(cov_opts.recenter)
    cov_opts.recenter = false;
end

cov_opts.type = char(lower(string(cov_opts.type)));
cov_opts.bandwidth_rule = char(lower(string(cov_opts.bandwidth_rule)));
cov_opts.bandwidth = double(cov_opts.bandwidth);
cov_opts.winsor_margin = double(cov_opts.winsor_margin);

end

function cov_opts = local_resolve_cov_bandwidth(cov_opts, T)
%% Convert an optional bandwidth rule into a numeric lag length.

cov_opts = local_default_cov_opts(cov_opts);
rule = char(lower(string(cov_opts.bandwidth_rule)));
if isempty(rule)
    return;
end

switch rule
    case 'nw_4_t100_2over9'
        cov_opts.bandwidth = bartlett_hac_bandwidth(T);
    case 't_1over4'
        cov_opts.bandwidth = floor(T^(1/4));
    case 't_1over3'
        cov_opts.bandwidth = floor(T^(1/3));
    otherwise
        error('run_bs_simulated_shape_rejection_study:BadBandwidthRule', ...
            'Unknown HAC bandwidth rule: %s.', rule);
end

end

function label = local_covariance_label(cov_opts)
%% Build a compact covariance label for output tags and tables.

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
