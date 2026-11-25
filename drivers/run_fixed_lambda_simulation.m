function outputs = run_fixed_lambda_simulation(stage, resume, lambdaValues, opts)
%% Fixed-lambda increasing-T simulations (BS or CHJ): one supplied lambda per run, no selection.
%
%   run_fixed_lambda_simulation('bs_increasing_t')                % all five design lambdas, resume on
%   run_fixed_lambda_simulation('chj_increasing_t', true, 1e-2)   % one lambda only
%
% Design (fixed_lambda_design below): BS lambda in {1e-6,1e-7,1e-8,1e-10,1e-12},
% CHJ lambda in {1e-1,1e-2,1e-3,1e-4,1e-6}; T in {500,1000,2000,5000,10000}; B = 1000;
% nMC = 10000. DGPs, seeds (the fixed per-T seed0 of the study designs, the same as in the
% CV and eff_dim runs), option and test grids, kernels, HAC and the Monte Carlo calibration
% are those of the main simulation studies: this runner only calls run_bs_study / run_chj_study with
% lambdaMethod 'fixed_lambda' and the supplied lambda. That lambda is used for the fit and
% for inference at every T and replication; no effective-dimension or CV routine runs.
%
% Output: results/simulations/fixed_lambda/<stage>/lambda_<tag>/ (block checkpoints under
% rejection_studies/, aggregated <bs|chj>_* files) and fixed_lambda_design.csv.
% Every (lambda, T) is checkpointed in replication blocks and resumes independently.
%
% opts (small test runs only): B, nMC, blockSize, useParallel, Tfilter (CHJ), requirePool.

setpath();
if nargin < 1 || isempty(stage), error('Specify stage: bs_increasing_t or chj_increasing_t.'); end
stage = validatestring(char(string(stage)), {'bs_increasing_t','chj_increasing_t'}, mfilename, 'stage');
if nargin < 2 || isempty(resume), resume = true; end
design = fixed_lambda_design(stage);
if nargin < 3 || isempty(lambdaValues), lambdaValues = design.lambda; end
if nargin < 4 || isempty(opts), opts = struct(); end
lambdaValues = double(lambdaValues(:).');
assert(all(ismember(lambdaValues, design.lambda)), 'run_fixed_lambda_simulation:OffDesign', ...
    'lambda values must come from the frozen %s grid %s.', stage, mat2str(design.lambda));
B = local_opt(opts, 'B', design.B);
nMC = local_opt(opts, 'nMC', design.nMC);
useParallel = local_opt(opts, 'useParallel', true);
production = B == design.B && nMC == design.nMC;

verify_simulation_design();
if useParallel && local_opt(opts, 'requirePool', production)
    local_require_process_pool(12);
end

root = fullfile(simulation_output_dir('simulations', 'fixed_lambda'), stage);
if ~isfolder(root), mkdir(root); end
nL = numel(design.lambda);
manifest = table(repmat(string(stage), nL, 1), design.lambda(:), ...
    string(arrayfun(@fixed_lambda_tag, design.lambda(:), 'UniformOutput', false)), ...
    repmat(string(mat2str(design.T)), nL, 1), repmat(B, nL, 1), repmat(nMC, nL, 1), ...
    'VariableNames', {'stage', 'lambda', 'folder', 'T_grid', 'B', 'nMC'});
writetable(manifest, fullfile(root, 'fixed_lambda_design.csv'));

fprintf('\nFIXED-LAMBDA SIMULATION: %s\n', stage);
fprintf('lambda values this call: %s (no selection; lambda_inference = lambda_fit = lambda)\n', ...
    mat2str(lambdaValues));
fprintf('T grid: %s, B = %d, nMC = %d, resume = %d\n', mat2str(design.T), B, nMC, resume);

outputs = struct('lambda', num2cell(lambdaValues), 'result', []);
for il = 1:numel(lambdaValues)
    lam = lambdaValues(il);
    fprintf('\n==================== %s: lambda = %g (%s) ====================\n', ...
        stage, lam, fixed_lambda_tag(lam));
    switch stage
        case 'bs_increasing_t'
            outputs(il).result = run_bs_study(B, local_opt(opts, 'blockSize', 25), nMC, useParallel, ...
                stage, resume, 'monthly_performance', 'fixed_lambda', [], lam);
        case 'chj_increasing_t'
            outputs(il).result = run_chj_study(B, local_opt(opts, 'blockSize', 12), nMC, useParallel, ...
                stage, resume, local_opt(opts, 'Tfilter', design.T), 'fixed_lambda', lam);
    end
end
fprintf('\nFIXED-LAMBDA SIMULATION COMPLETED: %s, lambda = %s\n', stage, mat2str(lambdaValues));

end

function d = fixed_lambda_design(stage)
%% The pre-specified fixed-lambda grids.
d = struct('T', [500 1000 2000 5000 10000], 'B', 1000, 'nMC', 10000);
switch stage
    case 'bs_increasing_t', d.lambda = [1e-6 1e-7 1e-8 1e-10 1e-12];
    case 'chj_increasing_t', d.lambda = [1e-1 1e-2 1e-3 1e-4 1e-6];
end
end

function v = local_opt(opts, name, default)
if isfield(opts, name) && ~isempty(opts.(name)), v = opts.(name); else, v = default; end
end

function pool = local_require_process_pool(nWorkers)
%% Same pool rule as run_simulation_production: exactly nWorkers process workers.
pool = gcp('nocreate');
if isempty(pool) || ~isa(pool, 'parallel.ProcessPool') || pool.NumWorkers ~= nWorkers
    if ~isempty(pool), delete(pool); end
    pool = parpool("Processes", nWorkers);
end
assert(isa(pool, 'parallel.ProcessPool') && pool.NumWorkers == nWorkers, ...
    'Fixed-lambda production requires exactly %d process workers.', nWorkers);
fprintf('Parallel pool: Processes, workers = %d\n', pool.NumWorkers);
end
