function summary_all = run_empirical_shape_tests(nMC, force_recompute, horizon_subset, lambdaMethod)
%% Run the empirical shape tests on the saved fitted states.
%
% The fitted states written by run_empirical_pipeline are immutable inputs:
% each saved coefficient vector is re-verified against its saved linear
% system before use, and no data generation, fitting, or CV is performed.
% Inference always tests the fitted object (lambda_inference = lambda_fit).
% Conditional VIX states are the five figure-selector states of the fitted
% model; the equal-date row averages the exogenous representer over observed
% per-date VIX clipped to the fitted exogenous support.
%
% lambdaMethod ('eff_dim' default, or 'cv') selects which saved fits are
% tested and where outputs go; the testing code is identical for both.
%
% Outputs (non-overwriting, checkpointed per maturity; locations resolved by
% config_empirical):
%   results/empirical/<lambda_method>/<horizon>/empirical_<horizon>.csv / .mat
%   results/empirical/<lambda_method>/empirical_all.csv

setpath();
design = get_frozen_design();
if nargin < 1 || isempty(nMC), nMC = design.nMC; end
if nargin < 2 || isempty(force_recompute), force_recompute = false; end
if nargin < 3 || isempty(horizon_subset)
    horizons = {'quarterly','monthly','weekly','daily'};
else
    if ischar(horizon_subset) || isstring(horizon_subset)
        horizon_subset = cellstr(string(horizon_subset));
    end
    validHorizons = {'quarterly','monthly','weekly','daily'};
    assert(iscell(horizon_subset) && all(ismember(horizon_subset, validHorizons)), ...
        'horizon_subset must contain quarterly, monthly, weekly, and/or daily.');
    horizons = horizon_subset(:).';
end
if nargin < 4, lambdaMethod = []; end
nMC = double(nMC);
force_recompute = logical(force_recompute);
assert(nMC >= 1 && nMC == floor(nMC), 'nMC must be a positive integer.');
assert(nMC <= design.nMC, 'nMC cannot exceed the frozen production value.');

% Parallel only when the toolbox is both licensed and installed. Each MC draw
% is seeded by its own index (build_mc_pvalue), so serial and parallel runs
% give identical statistics and p-values.
useParallel = license('test', 'Distrib_Computing_Toolbox') && exist('gcp', 'file') == 2;
if useParallel
    pool = gcp('nocreate');
    if isempty(pool), parpool; end
end

fprintf('\nEMPIRICAL SHAPE TESTS\n');
fprintf('inference rule: lambda_inference = lambda_fit\n');
fprintf('nMC           : %d\n\n', nMC);

for ih = 1:numel(horizons)
    horizon = horizons{ih};
    cfg = config_empirical(horizon, lambdaMethod);
    cfg.testing.nMC = nMC;
    methodPath = cfg.results.methodPath;

    modelFile = cfg.dataset.modelStateFile;
    assert(isfile(modelFile), 'Missing saved fitted state: %s', modelFile);
    S = load(modelFile, 'model_state');
    assert(isfield(S, 'model_state'), 'Malformed saved model file: %s', modelFile);
    fit_state = S.model_state;
    verify_saved_fit(fit_state, cfg, horizon);
    if strcmp(cfg.lambda_method, 'eff_dim')
        % Score covariances use the QR factor of the same system (in memory
        % only; the saved fitted state is not modified).
        [~, fit_state.solve_factor] = solve_projection_qr( ...
            build_payoff_factor(fit_state.alpha), fit_state.K_h, fit_state.lambda);
    end

    [volLevels, selInfo] = select_vix_levels(cfg, fit_state);
    assert(numel(unique(round(100 * volLevels))) == numel(volLevels), ...
        '%s conditional VIX states collide under the MC seed key.', horizon);
    sel = struct('lambda_fit', fit_state.lambda, ...
        'lambda_inference', fit_state.lambda, 'rule', design.inference_rule);
    lambda_selection = struct('lambda_method', cfg.lambda_method, ...
        'lambda_fit', fit_state.lambda, 'lambda_inference', fit_state.lambda);
    if strcmp(cfg.lambda_method, 'eff_dim')
        lambda_selection.effdim_c = fit_state.effdim.c;
        lambda_selection.effdim_target = fit_state.effdim.target;
        lambda_selection.effdim_residual = fit_state.effdim.residual;
        lambda_selection.numerical_rank = fit_state.effdim.numerical_rank;
    end

    design_fingerprint = build_empirical_design_fingerprint(cfg, fit_state, nMC);
    design_fingerprint.conditional_vix_states = volLevels;

    df = readtable(cfg.dataset.dfFile, 'VariableNamingRule', 'preserve');
    df = sortrows(df, {'date','frwd_moneyness'});
    [~, firstIdx] = unique(df.date, 'stable');
    dateList = df.date(firstIdx);
    T = numel(dateList);
    assert(fit_state.T == T, '%s saved T does not match the panel.', horizon);

    covOpts = postfit_hac_covariance_options(T);
    grids = build_region_grids(cfg);
    assert(numel(grids) == 3 && sum(arrayfun(@(g) numel(g.m), grids)) == 60, ...
        '%s shape-test grid changed.', horizon);

    if ~isfolder(cfg.results.path), mkdir(cfg.results.path); end
    checkpointFile = fullfile(cfg.results.path, ['empirical_' horizon '.mat']);
    checkpointCsv = fullfile(cfg.results.path, ['empirical_' horizon '.csv']);

    reuseCheckpoint = false;
    if isfile(checkpointFile) && ~force_recompute
        C = load(checkpointFile, 'summary_h', 'nMC_saved', 'design_fingerprint');
        assert(C.nMC_saved == nMC, '%s checkpoint nMC mismatch.', horizon);
        assert_design_fingerprint(C.design_fingerprint, design_fingerprint);
        reuseCheckpoint = true;
        fprintf('Reusing completed checkpoint: %s\n', horizon);
    end

    if ~reuseCheckpoint
        rows = cell(numel(volLevels)+1, 1);
        for iv = 1:numel(volLevels)
            exoOpts = struct('mode','conditional','value',volLevels(iv));
            exo = build_exo_representer(cfg, fit_state, df, exoOpts);
            rows{iv} = local_state_tests(cfg, fit_state, dateList, grids, exo, ...
                'conditional', volLevels(iv), covOpts, nMC, useParallel, ...
                horizon, sel);
        end
        exoOpts = struct('mode','equal_date','clip_to_grid',true);
        exo = build_exo_representer(cfg, fit_state, df, exoOpts);
        rows{end} = local_state_tests(cfg, fit_state, dateList, grids, exo, ...
            'equal_date', NaN, covOpts, nMC, useParallel, horizon, sel);

        summary_h = cat(1, rows{:});
        summary_h = sortrows(summary_h, {'mode','exo_level','region_order','test_order'});
        nMC_saved = nMC;
        vix_selection = selInfo;
        save(checkpointFile, 'summary_h','cfg','modelFile','nMC_saved', ...
            'design_fingerprint','vix_selection','lambda_selection','-v7.3');
        writetable(summary_h, checkpointCsv);
    end

    fprintf('%s [%s]: lambda_fit = lambda_inference = %.9g | VIX states %s\n', ...
        upper(horizon), cfg.lambda_method, fit_state.lambda, ...
        strjoin(compose('%.4f', volLevels), ', '));
end

summary_all = local_collect_completed_horizons(methodPath);
writetable(summary_all, fullfile(methodPath, 'empirical_all.csv'));
end

function summaryAll = local_collect_completed_horizons(methodPath)
%% Combine every completed maturity checkpoint without recomputation.
% Per-maturity checkpoints live in methodPath/<horizon>/ (config_empirical).

horizons = {'quarterly','monthly','weekly','daily'};
summaryCells = cell(0,1);
for j = 1:numel(horizons)
    file = fullfile(methodPath, horizons{j}, ['empirical_' horizons{j} '.mat']);
    if isfile(file)
        S = load(file, 'summary_h');
        summaryCells{end+1,1} = S.summary_h; %#ok<AGROW>
    end
end
assert(~isempty(summaryCells), 'No completed empirical shape-test checkpoints found.');
summaryAll = cat(1, summaryCells{:});

end

function rows = local_state_tests(cfg, model_state, dateList, grids, exo, ...
    mode, exoLevel, covOpts, nMC, useParallel, horizon, sel)
%% Run mono/conv/conc cone tests for one exogenous state across all regions.

T = model_state.T;
nRegions = numel(grids);
monoTheta = cell(nRegions,1);
convTheta = cell(nRegions,1);
monoIdx = cell(nRegions,1);
curvIdx = cell(nRegions,1);
Lcells = cell(2*nRegions,1);
col0 = 0;

for ir = 1:nRegions
    [theta,L] = build_theta_hat(cfg, model_state, grids(ir), 'mono', exo);
    monoTheta{ir} = theta;
    Lcells{ir} = L;
    idx = col0 + (1:numel(theta));
    monoIdx{ir} = idx;
    col0 = idx(end);
end
for ir = 1:nRegions
    [theta,L] = build_theta_hat(cfg, model_state, grids(ir), 'conv', exo);
    convTheta{ir} = theta;
    Lcells{nRegions+ir} = L;
    idx = col0 + (1:numel(theta));
    curvIdx{ir} = idx;
    col0 = idx(end);
end
Lall = cat(2, Lcells{:});

[~, Sall, omegaInfo] = build_Omega_hat_ls(model_state, Lall);
[OmegaAll, ~, covInfo] = build_score_covariance(Sall, dateList, covOpts);

out = cell(3*nRegions,1);
k = 0;
for ir = 1:nRegions
    grid = grids(ir);

    thetaMono = monoTheta{ir};
    OmegaMono = OmegaAll(monoIdx{ir},monoIdx{ir});
    OmegaMono = 0.5*(OmegaMono+OmegaMono.');
    statMono = solve_wald_projection(thetaMono,OmegaMono,[],T);
    mcOpts = struct('parallel',useParallel, ...
        'rng_seed',local_seed(horizon,mode,exoLevel,ir,'mono'));
    [pMono,~,Wmono,metaMono] = build_mc_pvalue( ...
        statMono.W_T,OmegaMono,[],T,cfg.testing.alpha,nMC,mcOpts);

    k = k+1;
    out{k} = local_row(horizon,mode,exoLevel,grid,ir,'mono',1, ...
        thetaMono,OmegaMono,statMono,pMono,Wmono,metaMono,covInfo,omegaInfo,nMC,sel);

    thetaConv = convTheta{ir};
    thetaConc = -thetaConv;
    OmegaCurv = OmegaAll(curvIdx{ir},curvIdx{ir});
    OmegaCurv = 0.5*(OmegaCurv+OmegaCurv.');
    statConv = solve_wald_projection(thetaConv,OmegaCurv,[],T);
    statConc = solve_wald_projection(thetaConc,OmegaCurv,[],T);
    mcOpts.rng_seed = local_seed(horizon,mode,exoLevel,ir,'curvature');
    [pConv,~,Wcurv,metaCurv] = build_mc_pvalue( ...
        statConv.W_T,OmegaCurv,[],T,cfg.testing.alpha,nMC,mcOpts);
    valid = Wcurv(isfinite(Wcurv));
    if isempty(valid)
        pConc = NaN;
    else
        pConc = (sum(valid >= statConc.W_T)+1)/(numel(valid)+1);
    end

    k = k+1;
    out{k} = local_row(horizon,mode,exoLevel,grid,ir,'conv',2, ...
        thetaConv,OmegaCurv,statConv,pConv,Wcurv,metaCurv,covInfo,omegaInfo,nMC,sel);
    k = k+1;
    out{k} = local_row(horizon,mode,exoLevel,grid,ir,'conc',3, ...
        thetaConc,OmegaCurv,statConc,pConc,Wcurv,metaCurv,covInfo,omegaInfo,nMC,sel);
end
rows = cat(1,out{:});
end

function row = local_row(horizon,mode,exoLevel,grid,regionOrder,testType,testOrder, ...
    theta,Omega,stat,pValue,Wmc,mcMeta,covInfo,omegaInfo,nMC,sel)
%% Assemble one inference row with the fit/inference lambdas explicit.

valid = Wmc(isfinite(Wmc));
if isempty(valid)
    crit1=NaN; crit5=NaN; crit10=NaN;
else
    crit1=quantile(valid,.99); crit5=quantile(valid,.95); crit10=quantile(valid,.90);
end
eigOmega = eig(0.5*(Omega+Omega.'));

row = table(string(horizon),string(mode),exoLevel,string(grid.name),regionOrder, ...
    grid.bounds(1),grid.bounds(2),string(testType),testOrder,omegaInfo.T, ...
    sel.lambda_fit,sel.lambda_inference,sel.lambda_inference/sel.lambda_fit, ...
    string(sel.rule), ...
    double(nMC),double(stat.W_T),double(pValue),crit5,crit10,crit1, ...
    double(pValue<.10),double(pValue<.05),double(pValue<.01), ...
    numel(theta),min(theta),max(theta),mean(theta<0), ...
    covInfo.bandwidth,string(covInfo.type),min(eigOmega),max(eigOmega), ...
    omegaInfo.condition_number,double(mcMeta.n_valid),double(mcMeta.n_failed), ...
    'VariableNames', {'horizon','mode','exo_level','region','region_order', ...
    'region_left','region_right','testType','test_order','T', ...
    'lambda_fit','lambda_inference','lambda_ratio','inference_rule', ...
    'nMC','W_T','p_value','critical_5pct','critical_10pct','critical_1pct', ...
    'reject_10pct','reject_5pct','reject_1pct','n_grid','min_theta','max_theta', ...
    'frac_negative','hac_bandwidth','covariance_type','omega_min_eig', ...
    'omega_max_eig','inference_condition_number','mc_n_valid','mc_n_failed'});
end

function seed = local_seed(horizon,mode,exoLevel,regionOrder,testGroup)
%% Deterministic Monte Carlo seed convention.

if isnan(exoLevel), exoCode=0; else, exoCode=round(100*exoLevel); end
key=[horizon '_' mode '_' testGroup];
seed=240000+10000*regionOrder+exoCode;
for j=1:numel(key), seed=seed+j*double(key(j)); end
seed=mod(seed,2^31-1);
end

function cov_opts = postfit_hac_covariance_options(T)
%% Bartlett HAC covariance options for post-fit inference.
cov_opts = struct();
cov_opts.type = 'hac_bartlett';
cov_opts.bandwidth = bartlett_hac_bandwidth(T);
cov_opts.bandwidth_rule = 'nw_4_t100_2over9';
cov_opts.winsor_margin = 0;
cov_opts.recenter = false;
end
