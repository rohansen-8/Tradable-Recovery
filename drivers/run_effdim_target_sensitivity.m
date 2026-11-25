function T = run_effdim_target_sensitivity(horizon, cs, doOOS)
%% Effective-dimension target sensitivity: N_T(lambda) = c*T^(1/3), c in {1/2, 1, 2} (IA table).
%
%   T = run_effdim_target_sensitivity()                    % monthly, c = [0.5 1 2], in-sample + OOS
%   T = run_effdim_target_sensitivity('monthly', 2, false)  % one target, in-sample only
%
% Only the target constant c changes; the estimator is train_projection_model
% and the OOS exercise is run_empirical_oos_attribution, as in the main results.
% Columns:
%   N_T, lambda    effective dimension at the selected lambda, lambda itself;
%   sr_is          annualized in-sample Sharpe ratio of alpha_t' h_star;
%   sr_oos         annualized Sharpe ratio of the OOS total return (primary rule,
%                  lambda re-solved in every window at the same c);
%   min_sdf        min over dates and the figure log range (200 points) of the
%                  fitted SDF M at the date's VIX (clipped to the fitted grid),
%                  divided by its overall mean.
% Outputs, refusing to overwrite: <results_root>/empirical_effdim_target/<h>/
%   c<c>/ (fit, OOS windows) and effdim_target_<h>.csv. c = 1 is the
% baseline rule (its lambda is checked against the saved baseline fit when present).
% Runtime (monthly): seconds per fit; ~3 min of OOS fits per c.

if nargin < 1 || isempty(horizon), horizon = 'monthly'; end
if nargin < 2 || isempty(cs), cs = [0.5 1 2]; end
if nargin < 3 || isempty(doOOS), doOOS = true; end
setpath();
outRoot = fullfile(results_root(), 'empirical_effdim_target', horizon);
csvOut = fullfile(outRoot, sprintf('effdim_target_%s.csv', horizon));
assert(~isfile(csvOut), 'run_effdim_target_sensitivity:Exists', 'Refusing to overwrite %s', csvOut);

base = config_empirical(horizon, 'eff_dim');
df = readtable(base.dataset.dfFile, 'VariableNamingRule', 'preserve');
m = get_empirical_maturity_config(horizon);
N = struct('quarterly', 4, 'monthly', 12, 'weekly', 52, 'daily', 252).(horizon);
rows = zeros(numel(cs), 6);
for i = 1:numel(cs)
    c = cs(i);
    d = fullfile(outRoot, sprintf('c%g', c));
    if ~isfolder(d), mkdir(d); end
    cfg = base;                                  % only the target constant and save paths change
    cfg.training.effdim.c = c;
    cfg.dataset.path = d; cfg.results.path = d; cfg.results.methodPath = d; cfg.results.figuresPath = d;
    cfg.dataset.modelStateFile = fullfile(d, sprintf('%s_model_state.mat', horizon));
    cfg.results.cvCheckpoint = fullfile(d, 'unused_cv.mat');
    ms = train_projection_model(cfg, df);
    s = ms.alpha.' * ms.h_star;
    if c == 1 && isfile(base.dataset.modelStateFile)
        B = load(base.dataset.modelStateFile, 'model_state');
        assert(abs(ms.lambda / B.model_state.lambda - 1) < 1e-10, 'baseline lambda not reproduced');
    end
    srOOS = NaN;
    if doOOS
        o = struct('mode', 'validation', 'outRoot', fullfile(d, 'oos'), 'effdim_c', c);
        run_empirical_oos_attribution({horizon}, o);
        A = assemble_oos_attribution(horizon, o);
        srOOS = mean(A.total) / std(A.total) * sqrt(N);
    end
    rows(i, :) = [c, ms.effdim.N_at_lambda, ms.lambda, mean(s) / std(s) * sqrt(N), srOOS, ...
        local_min_sdf(cfg, ms, df, m.figure_log_range)];
    fprintf('c=%-4g N_T=%6.2f lambda=%.4g SR_is=%.2f SR_oos=%.2f minSDF=%.2f\n', rows(i, :));
end
T = array2table(rows, 'VariableNames', {'c','N_T','lambda','sr_is','sr_oos','min_sdf'});
writetable(T, csvOut);
end

function v = local_min_sdf(cfg, ms, df, logRange)
%% Smallest mean-one-normalized fitted SDF over dates (own VIX, clipped) and the figure range.
dates = df.date; if ~isdatetime(dates), dates = datetime(string(dates)); end
[~, first] = unique(dates);
g = ms.exo_grid(:);
vix = min(max(df.VIX(first), min(g)), max(g));
[u, ~, k] = unique(vix);
mRaw = exp(linspace(logRange(1), logRange(2), 200)).';
M = zeros(numel(u), numel(mRaw));
for j = 1:numel(u)
    exo = build_exo_representer(cfg, ms, [], struct('mode', 'conditional', 'value', u(j)));
    out = evaluate_sdf_grid(cfg, ms, mRaw, exo);
    M(j, :) = out.M(:).';
end
M = M(k, :);                                     % one row per date
v = min(M(:) / mean(M(:)));
end
