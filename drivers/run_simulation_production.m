function outputs = run_simulation_production(stage, resume, lambdaMethod, horizon)
%% Run one simulation stage of the final design with 12 process workers.
%
% lambdaMethod: 'eff_dim' (default, get_frozen_design().
% simulation_lambda_method; results under results/simulations/eff_dim/) or
% 'cv' (the CV robustness results under results/simulations/cv/).
%
% horizon (bs_empirical_t only): one maturity, 'quarterly', 'monthly',
% 'weekly' or 'daily'. Under eff_dim it is required, so no maturity is ever
% launched implicitly; each maturity runs and saves on its own
% (bs_results_<horizon>.mat). Under cv, omitting it runs all
% four maturities in one call. If a daily replication's payoff rank does not exceed
% the effective-dimension target, the run stops with
% select_lambda_effective_dimension:Unattainable naming the seed.
%
% bs_increasing_t and chj_increasing_t are independent studies and may be
% run in either order. bs_empirical_t reuses the bs_increasing_t design point
% ordering and still runs after it.
%
% Every replication selects lambda_fit by the chosen rule (effective
% dimension or one-step CV). Shape inference uses lambda_inference =
% lambda_fit; there is no separate post-fit regularization selection.

setpath();
if nargin < 1 || isempty(stage)
    error('Specify stage: bs_increasing_t, bs_empirical_t, or chj_increasing_t.');
end
if nargin < 2 || isempty(resume), resume = true; end
if nargin < 3 || isempty(lambdaMethod)
    lambdaMethod = get_frozen_design().simulation_lambda_method;
end
if nargin < 4, horizon = []; end
stage = lower(string(stage));
resume = logical(resume);
lambdaMethod = validatestring(char(string(lambdaMethod)), {'cv','eff_dim'}, ...
    mfilename, 'lambdaMethod');
fprintf('Simulation lambda method: %s\n', lambdaMethod);
if stage == "bs_empirical_t"
    if isempty(horizon)
        assert(strcmp(lambdaMethod, 'cv'), 'run_simulation_production:HorizonRequired', ...
            ['bs_empirical_t under eff_dim runs one maturity per call: ' ...
             'run_simulation_production(''bs_empirical_t'', true, ''eff_dim'', ''quarterly'') ' ...
             '(or ''monthly'', ''weekly'', ''daily'').']);
    else
        horizon = string(validatestring(char(string(horizon)), ...
            {'quarterly','monthly','weekly','daily'}, mfilename, 'horizon'));
    end
else
    assert(isempty(horizon), 'run_simulation_production:BadHorizon', ...
        'horizon applies to bs_empirical_t only.');
end

verify_simulation_design();
audit_simulation_efficiency(lambdaMethod);
local_require_process_pool(12);

% B is the number of independent simulated datasets per design point and is
% read from the design, not fixed here: the two increasing-T studies
% keep B = 1000 and bs_empirical_t uses B = 2000. nMC is the separate
% Gaussian Monte Carlo count used inside each shape test.
nMC = 10000;
useParallel = true;
outputs = struct();

switch stage
    case "bs_increasing_t"
        cfg = config_bs_simulation_design('monthly', 'bs_increasing_t', lambdaMethod);
        outputs.BS = run_bs_study(cfg.sim.final_B, 25, nMC, useParallel, ...
            'bs_increasing_t', resume, 'monthly_performance', lambdaMethod);

    case "bs_empirical_t"
        local_require_completed_bs_stage('bs_increasing_t', lambdaMethod);
        cfg = config_bs_simulation_design('quarterly', 'bs_empirical_t', lambdaMethod);
        outputs.BS = run_bs_study(cfg.sim.final_B, 25, nMC, useParallel, ...
            'bs_empirical_t', resume, 'empirical_T', lambdaMethod, horizon);

    case "chj_increasing_t"
        cfg = config_chj_simulation_design(lambdaMethod);
        outputs.CHJ = run_chj_study(cfg.sim.final_B, 12, nMC, useParallel, ...
            'chj_increasing_t', resume, cfg.sim.performance_T, lambdaMethod);

    otherwise
        error('Unknown final simulation stage %s.', stage);
end

if isempty(horizon)
    fprintf('\nFINAL STAGE COMPLETED: %s\n', stage);
else
    fprintf('\nFINAL STAGE COMPLETED: %s (%s)\n', stage, horizon);
end

end

function local_require_completed_bs_stage(tag, lambdaMethod)
%% Require the completed BS increasing-T stage before bs_empirical_t.

file = fullfile(simulation_output_dir('simulations', lambdaMethod), tag, ...
    'bs_results.mat');
assert(isfile(file), ...
    'Required prior BS stage is not complete: %s', tag);

end

function pool = local_require_process_pool(nWorkers)
%% Use the requested process pool so v7.3 checkpoints remain loadable.

pool = gcp('nocreate');
needsNewPool = isempty(pool) || ~isa(pool, 'parallel.ProcessPool') ...
    || pool.NumWorkers ~= nWorkers;
if needsNewPool
    if ~isempty(pool), delete(pool); end
    pool = parpool("Processes", nWorkers);
end
assert(isa(pool, 'parallel.ProcessPool') && pool.NumWorkers == nWorkers, ...
    'Final production requires exactly %d process workers.', nWorkers);
fprintf('Parallel pool: Processes, workers = %d\n', pool.NumWorkers);

end
