function test_empirical_lambda_switch()
%% Switch and downstream-interface checks on a small synthetic panel.
%
% Builds the real monthly configuration under both lambda methods, redirects
% every data and output path to a temporary folder, and fits a synthetic
% 80-date panel through the production train_projection_model. Checks:
%   - config_empirical: default 'eff_dim', method-specific paths, identical
%     non-lambda settings, identical CV block, invalid methods rejected;
%   - eff_dim fit: rule satisfied, no CV run, metadata recorded, direct
%     T-by-T Gram check on the real alpha panel;
%   - cv fit: no lambda_method/effdim fields, lambda on the fixed grid;
%   - downstream: verify_saved_fit, method-specific fingerprints, the CV
%     fingerprint field set equals that of the saved CV fit, cross-method resume
%     refused, select_vix_levels runs on both fits.
% No empirical data are read and nothing is written under results/.

setpath();
design = get_frozen_design();

%% Configuration switch
cfgE = config_empirical('monthly');
cfgC = config_empirical('monthly', 'cv');
assert(strcmp(design.empirical_lambda_method, 'eff_dim') && strcmp(cfgE.lambda_method, 'eff_dim'));
assert(strcmp(cfgC.lambda_method, 'cv'));
assert(endsWith(cfgE.results.path, fullfile('results','empirical','eff_dim','monthly')));
assert(endsWith(cfgC.results.path, fullfile('results','empirical','cv','monthly')));
assert(endsWith(cfgE.dataset.modelStateFile, fullfile('eff_dim','monthly','monthly_model_state.mat')));
assert(~cfgE.training.use_cv && cfgC.training.use_cv);
assert(isequal(cfgE.training.cv, cfgC.training.cv), 'CV specification must not depend on the switch.');
assert(isequal(cfgC.training.cv.lambda_grid(:), logspace(-5,-2,31).'));
strip = @(c) rmfield(rmfield(c, {'lambda_method','results','dataset'}), 'training');
assert(isequal(strip(cfgE), strip(cfgC)), 'Only lambda selection and paths may differ.');
trE = rmfield(cfgE.training, {'lambda_method','use_cv'});
trC = rmfield(cfgC.training, {'lambda_method','use_cv'});
assert(isequal(trE, trC));
expect_error(@() config_empirical('monthly', 'grid'), 'MATLAB:config_empirical:unrecognizedStringChoice');
fprintf('  config switch: default eff_dim, method paths, shared settings OK\n');

%% Synthetic panel in a temporary folder
tmp = tempname; mkdir(tmp); cleanup = onCleanup(@() rmdir(tmp, 's'));
df = local_synthetic_panel(80, 15, cfgE.training.mon_domain);
dfFile = fullfile(tmp, 'df.csv');
writetable(df, dfFile);

fits = struct();
for method = {'eff_dim', 'cv'}
    m = method{1};
    cfg = config_empirical('monthly', m);
    cfg.dataset.dfFile = dfFile;
    cfg.dataset.path = fullfile(tmp, m);
    cfg.results.path = cfg.dataset.path;
    cfg.dataset.modelStateFile = fullfile(cfg.dataset.path, 'model_state.mat');
    cfg.training.cv.min_train_dates = 60;          % 20 CV folds on 80 dates
    fits.(m).cfg = cfg;
    fits.(m).ms = train_projection_model(cfg, df);
    assert(isfile(cfg.dataset.modelStateFile));
end

%% eff_dim fit
ms = fits.eff_dim.ms; cfg = fits.eff_dim.cfg;
assert(strcmp(ms.lambda_method, 'eff_dim') && isempty(ms.cv));
assert(ms.lambda == ms.effdim.lambda && abs(ms.effdim.residual) <= 1e-10 * ms.effdim.target);
assert(abs(ms.effdim.target - ms.T^(1/3)) <= 1e-14 * ms.T^(1/3));
assert(ms.effdim.numerical_rank <= min(numel(ms.h_star), ms.T), 'Rank exceeds min(Nh, T).');
assert(strcmp(ms.meta.lambda_selection.method, "eff_dim"));
G = ms.alpha.' * (ms.K_h \ ms.alpha);  G = 0.5 * (G + G.');
Ndirect = trace(G / (G + ms.lambda * ms.T * eye(ms.T)));
assert(abs(Ndirect - ms.T^(1/3)) <= 1e-6 * ms.T^(1/3), ...
    'Direct N_T on the synthetic alpha panel: %.10g vs %.10g.', Ndirect, ms.T^(1/3));
verify_saved_fit(ms, cfg, 'eff_dim synthetic');
fpE = build_empirical_design_fingerprint(cfg, ms, design.nMC);
assert(strcmp(fpE.lambda_method, 'eff_dim') && ~isfield(fpE, 'lambda_grid') && ~isfield(fpE, 'cv_method'));
[levelsE, ~] = select_vix_levels(cfg, ms);
assert(numel(levelsE) == design.vix_selector.n_levels);
fprintf(['  eff_dim fit: T = %d, lambda = %.6e, N - target = %.2e, direct N - target = %.2e, ' ...
    'rank = %d (<= T), discarded eig ratio = %.1e, rcond(K_h) = %.2e\n'], ms.T, ms.lambda, ...
    ms.effdim.residual, Ndirect - ms.T^(1/3), ms.effdim.numerical_rank, ...
    ms.effdim.discarded_eigenvalue_ratio, ms.effdim.rcond_K_h);

%% cv fit
msC = fits.cv.ms; cfgCv = fits.cv.cfg;
assert(~isfield(msC, 'lambda_method') && ~isfield(msC, 'effdim') && ~isempty(msC.cv));
assert(any(msC.lambda == cfgCv.training.cv.lambda_grid));
verify_saved_fit(msC, cfgCv, 'cv synthetic');
fpC = build_empirical_design_fingerprint(cfgCv, msC, design.nMC);
legacyFile = fullfile(get_project_root(), 'results', 'empirical', 'cv', 'monthly', 'monthly_model_state.mat');
if isfile(legacyFile)
    L = load(legacyFile, 'model_state');
    legacyFields = setdiff(fieldnames(L.model_state.design_fingerprint), {'conditional_vix_states'});
    assert(isequal(sort(fieldnames(fpC)), sort(legacyFields)), ...
        'CV fingerprint fields must equal the legacy saved fingerprint fields.');
    fprintf('  cv fingerprint: field set identical to saved legacy CV fingerprint\n');
end
[levelsC, ~] = select_vix_levels(cfgCv, msC);
assert(numel(levelsC) == design.vix_selector.n_levels);
fprintf('  cv fit: lambda = %.6g (grid point), %d CV folds\n', msC.lambda, size(msC.cv.loss_by_date, 1));

%% Cross-method resume refused
expect_error(@() build_empirical_design_fingerprint(cfgCv, ms, design.nMC), ...
    'build_empirical_design_fingerprint:MethodMismatch');
expect_error(@() build_empirical_design_fingerprint(cfg, msC, design.nMC), ...
    'build_empirical_design_fingerprint:MethodMismatch');
try
    verify_saved_fit(ms, cfgCv, 'cross'); error('cross-method verify passed');
catch err
    assert(contains(err.message, 'was not selected by lambda_method'));
end
fprintf('  cross-method fingerprints and verification refused\n');
disp('test_empirical_lambda_switch passed.');
end

function df = local_synthetic_panel(T, J, monDomain)
%% Small synthetic option panel with the columns the estimator reads.
rng(7, 'twister');
date = repelem((1:T).', J);
m = repmat(linspace(monDomain(1) + 0.02, monDomain(2) - 0.02, J).', T, 1);
vixDate = 12 + 25 * rand(T, 1);
VIX = repelem(vixDate, J);
retDate = 0.01 + 0.05 * randn(T, 1);
SP500_ret = repelem(retDate, J);
R = 1 + SP500_ret;
isPut = m < 1;
payoff = isPut .* max(m - R, 0) + ~isPut .* max(R - m, 0);
price = 0.01 + 0.2 * abs(m - 1) .^ 2 + 0.02;
OPT_NET_R_PAYOFF = payoff ./ price - 1;
weight = ones(T * J, 1);
frwd_moneyness = m;
df = table(date, frwd_moneyness, OPT_NET_R_PAYOFF, weight, SP500_ret, VIX);
end

function expect_error(fn, id)
try
    fn();
catch err
    assert(strcmp(err.identifier, id), 'Expected %s, got %s: %s', id, err.identifier, err.message);
    return
end
error('Expected error %s was not raised.', id);
end
