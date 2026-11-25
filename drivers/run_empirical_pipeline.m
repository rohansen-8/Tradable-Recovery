function outputs = run_empirical_pipeline(horizonSubset, resume, lambdaMethod)
%% Fit the empirical estimator and run the pre-specified shape tests.
%
% lambdaMethod selects lambda: 'eff_dim' (default, the effective-dimension
% rule N_T(lambda) = T^(1/3)) or 'cv' (one-step-ahead CV over the fixed
% lambda grid). Both share the estimator, the saved-state format and the shape tests;
% outputs go to results/empirical/<lambdaMethod>/ (config_empirical).
%
% Fitting is resumable: an existing saved state is verified against the
% design fingerprint and reused rather than refitted. Completed fits then
% feed run_empirical_shape_tests.

setpath();
if nargin < 1 || isempty(horizonSubset)
    horizons = {'quarterly','monthly','weekly','daily'};
elseif ischar(horizonSubset) || isstring(horizonSubset)
    horizons = cellstr(string(horizonSubset));
else
    horizons = horizonSubset;
end
if nargin < 2 || isempty(resume), resume = true; end
if nargin < 3, lambdaMethod = []; end
validHorizons = {'quarterly','monthly','weekly','daily'};
assert(iscell(horizons) && all(ismember(horizons, validHorizons)), ...
    'horizonSubset contains an unsupported maturity.');
resume = logical(resume);

design = get_frozen_design();
modelFiles = strings(numel(horizons),1);
fitStatus = strings(numel(horizons),1);
for j = 1:numel(horizons)
    horizon = horizons{j};
    cfg = config_empirical(horizon, lambdaMethod);
    modelFile = cfg.dataset.modelStateFile;
    if ~isfolder(cfg.results.path), mkdir(cfg.results.path); end

    if isfile(modelFile)
        assert(resume, 'A fit already exists for %s; pass resume=true.', horizon);
        S = load(modelFile, 'model_state');
        model_state = S.model_state;
        fitStatus(j) = "reused";
    else
        df = readtable(cfg.dataset.dfFile, 'VariableNamingRule', 'preserve');
        model_state = train_projection_model(cfg, df);
        fitStatus(j) = "fit";
    end

    design_fingerprint = build_empirical_design_fingerprint( ...
        cfg, model_state, design.nMC);
    if isfield(model_state, 'design_fingerprint')
        saved = rmfield_if_present(model_state.design_fingerprint, ...
            'conditional_vix_states');
        assert_design_fingerprint(saved, design_fingerprint);
    else
        model_state.design_fingerprint = design_fingerprint;
        save(modelFile, 'model_state', '-v7.3');
    end
    modelFiles(j) = string(modelFile);
end

summary = run_empirical_shape_tests(design.nMC, false, horizons, cfg.lambda_method);
outputs = struct('summary', summary, 'horizons', {horizons}, ...
    'lambda_method', cfg.lambda_method, ...
    'model_files', modelFiles, 'fit_status', fitStatus, 'resume', resume);

end

function s = rmfield_if_present(s, name)
%% Remove one struct field when it exists.

if isfield(s, name)
    s = rmfield(s, name);
end

end
