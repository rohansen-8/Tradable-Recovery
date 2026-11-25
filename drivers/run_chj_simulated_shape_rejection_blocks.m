function [summary_all, raw_all, block_log, regularization_bias_all, recovery_all] = run_chj_simulated_shape_rejection_blocks(Tvals, Btotal, blockSize, lambda, seed0, nMC, tagPrefix, useParallel, cov_opts, resume, lambdaMethod)
%% Run CHJ study simulations in restartable replication blocks.
%
% Blocks are computational checkpoints only. Each replication retains its own
% simulated state path, lengthscales, selected lambda, equal-date VIX
% representer, HAC covariance, and prespecified regularization-path diagnostic.
% lambdaMethod ('cv' default, or 'eff_dim') selects the fitting-lambda rule
% and the results namespace; checkpoints are fingerprinted per method.

setpath();
if nargin < 11 || isempty(lambdaMethod), lambdaMethod = 'cv'; end
cfg0 = config_chj_simulation_design(lambdaMethod);

if nargin < 1 || isempty(Tvals), Tvals = cfg0.sim.performance_T(1); end
if nargin < 2 || isempty(Btotal), Btotal = cfg0.sim.final_B; end
if nargin < 3 || isempty(blockSize), blockSize = 12; end
if nargin < 4, lambda = cfg0.training.lambda; end
if nargin < 5 || isempty(seed0), seed0 = cfg0.sim.seed0; end
if nargin < 6 || isempty(nMC), nMC = cfg0.testing.nMC; end
if nargin < 7 || isempty(tagPrefix), tagPrefix = 'chj_study'; end
if nargin < 8 || isempty(useParallel), useParallel = true; end
if nargin < 9 || isempty(cov_opts), cov_opts = empirical_covariance(cfg0); end
if nargin < 10 || isempty(resume), resume = true; end

Tvals = Tvals(:).';
Btotal = double(Btotal);
blockSize = double(blockSize);
if ~isempty(lambda), lambda = double(lambda); end
seed0 = double(seed0);
nMC = double(nMC);
tagPrefix = char(tagPrefix);
useParallel = logical(useParallel);
resume = logical(resume);
cov_opts = local_default_cov_opts(cov_opts);
covLabel = local_covariance_label(cov_opts);
revision_id = cfg0.revision.id;
regularizationLambdaPath = cfg0.diagnostics.regularization_lambda_path(:);
% fixed_lambda: bind the supplied lambda (fingerprint, lambda_<tag> results folder).
cfg0 = apply_fixed_lambda(cfg0, lambda);
design_fingerprint = build_design_fingerprint(cfg0, Tvals, Btotal, nMC);

assert(Btotal >= 1 && Btotal == floor(Btotal), 'Btotal must be a positive integer.');
assert(blockSize >= 1 && blockSize == floor(blockSize), 'blockSize must be a positive integer.');
assert(nMC >= 1 && nMC == floor(nMC), 'nMC must be a positive integer.');

nBlocks = ceil(Btotal / blockSize);

fprintf('\nCHJ study block run\n');
fprintf('  T values: %s\n', mat2str(Tvals));
fprintf('  B total per T: %d\n', Btotal);
fprintf('  block size: %d\n', blockSize);
fprintf('  number of blocks: %d\n', nBlocks);
if isempty(lambda) && strcmp(cfg0.training.lambda_method, 'eff_dim')
    fprintf('  lambda: effective dimension N_T(lambda) = %g * T^(1/3) (no CV)\n', ...
        cfg0.training.effdim_c);
elseif isempty(lambda)
    fprintf('  lambda: one-step-ahead CV on 31-point grid\n');
else
    fprintf('  lambda: fixed override %.6g\n', lambda);
end
fprintf('  regularization diagnostic path: %s\n', mat2str(regularizationLambdaPath.'));
fprintf('  nMC: %d\n', nMC);
fprintf('  seed0: %d\n', seed0);
fprintf('  useParallel: %d\n', useParallel);
fprintf('  covariance: %s\n', covLabel);
fprintf('  revision: %s\n', revision_id);
fprintf('  resume: %d\n\n', resume);

raw_cells = cell(nBlocks, 1);
reg_cells = cell(nBlocks, 1);
recovery_cells = cell(nBlocks, 1);
log_cells = cell(nBlocks, 1);
done = 0;

for ib = 1:nBlocks
    Bblock = min(blockSize, Btotal - done);
    blockSeed0 = seed0 + done;
    blockTag = sprintf('%s_block%02d', tagPrefix, ib);

    fprintf('\n============================================================\n');
    fprintf('Starting CHJ block %d / %d\n', ib, nBlocks);
    fprintf('  B in block per T: %d\n', Bblock);
    fprintf('  completed before block: %d / %d\n', done, Btotal);
    fprintf('  seed0 for block: %d\n', blockSeed0);
    fprintf('  tag: %s\n', blockTag);
    fprintf('============================================================\n');

    blockDir = fullfile(cfg0.results.path, 'rejection_studies', blockTag);
    blockFile = fullfile(blockDir, sprintf('%s.mat', blockTag));
    loadedCheckpoint = false;

    if resume && isfile(blockFile)
        saved = load(blockFile, 'raw', 'regularization_bias_raw', 'recovery_raw', ...
            'regularizationLambdaPath', 'Tvals', 'B', 'lambda', 'seed0', ...
            'nMC', 'cov_opts', 'revision_id', 'design_fingerprint');
        local_validate_checkpoint(saved, Tvals, Bblock, lambda, blockSeed0, ...
            nMC, cov_opts, revision_id, regularizationLambdaPath, design_fingerprint);
        raw_block = saved.raw;
        reg_block = saved.regularization_bias_raw;
        recovery_block = saved.recovery_raw;
        summary_block = summarize_chj_rejection_results(raw_block, cfg0.training.cv.lambda_grid);
        elapsed_block = 0;
        loadedCheckpoint = true;
        fprintf('  loaded completed block from checkpoint\n');
    else
        tBlock = tic;
        [summary_block, raw_block, reg_block, recovery_block] = run_chj_simulated_shape_rejection_study( ...
            Tvals, Bblock, lambda, blockSeed0, nMC, blockTag, true, ...
            useParallel, cov_opts, Btotal, lambdaMethod);
        elapsed_block = toc(tBlock);
    end

    raw_cells{ib} = raw_block;
    reg_cells{ib} = reg_block;
    recovery_cells{ib} = recovery_block;
    log_cells{ib} = table(ib, nBlocks, Bblock, blockSeed0, string(blockTag), ...
        elapsed_block, height(raw_block), height(reg_block), height(recovery_block), loadedCheckpoint, ...
        'VariableNames', {'block', 'nBlocks', 'Bblock', 'seed0', 'tag', ...
        'elapsed_seconds', 'n_raw_rows', 'n_regularization_rows', ...
        'n_recovery_rows', 'loaded_checkpoint'});

    done = done + Bblock;

    fprintf('\nFinished CHJ block %d / %d\n', ib, nBlocks);
    fprintf('  elapsed seconds: %.1f\n', elapsed_block);
    fprintf('  cumulative completed per T: %d / %d\n', done, Btotal);
    disp(summary_block(:, {'Tval', 'B', 'region', 'testType', 'role', ...
        'lambda_fit_median', 'lambda_inference_median', 'lambda_changed_freq', ...
        'ls_exo_median', 'exo_frac_clipped_mean', ...
        'rej_1pct', 'rej_5pct', 'rej_10pct'}));
end

raw_all = cat(1, raw_cells{:});
regularization_bias_all = local_cat_nonempty_tables(reg_cells);
recovery_all = local_cat_nonempty_tables(recovery_cells);
block_log = cat(1, log_cells{:});
summary_all = summarize_chj_rejection_results(raw_all, cfg0.training.cv.lambda_grid);
regularization_bias_summary = summarize_regularization_sensitivity(regularization_bias_all);
recovery_summary = summarize_sdf_recovery(recovery_all);

%% Save combined checkpoint output
outDir = fullfile(cfg0.results.path, 'rejection_studies', tagPrefix);
if ~isfolder(outDir), mkdir(outDir); end

rawFile = fullfile(outDir, sprintf('raw_%s_combined.csv', tagPrefix));
summaryFile = fullfile(outDir, sprintf('summary_%s_combined.csv', tagPrefix));
regRawFile = fullfile(outDir, sprintf('regularization_bias_%s_combined.csv', tagPrefix));
regSummaryFile = fullfile(outDir, sprintf('regularization_bias_summary_%s_combined.csv', tagPrefix));
recoveryRawFile = fullfile(outDir, sprintf('recovery_%s_combined.csv', tagPrefix));
recoverySummaryFile = fullfile(outDir, sprintf('recovery_summary_%s_combined.csv', tagPrefix));
blockLogFile = fullfile(outDir, sprintf('block_log_%s.csv', tagPrefix));
matFile = fullfile(outDir, sprintf('%s_combined.mat', tagPrefix));

% Every checkpoint write is atomic: the covariance rule, block size and
% design fingerprint live inside the saved objects, not in the filename.
local_atomic_writetable(raw_all, rawFile);
local_atomic_writetable(summary_all, summaryFile);
if ~isempty(regularization_bias_all), local_atomic_writetable(regularization_bias_all, regRawFile); end
if ~isempty(regularization_bias_summary), local_atomic_writetable(regularization_bias_summary, regSummaryFile); end
if ~isempty(recovery_all), local_atomic_writetable(recovery_all, recoveryRawFile); end
if ~isempty(recovery_summary), local_atomic_writetable(recovery_summary, recoverySummaryFile); end
local_atomic_writetable(block_log, blockLogFile);
matTmp = [matFile '.tmp.mat'];
if isfile(matTmp), delete(matTmp); end
save(matTmp, 'summary_all', 'raw_all', 'regularization_bias_all', ...
    'regularization_bias_summary', 'recovery_all', 'recovery_summary', ...
    'regularizationLambdaPath', 'block_log', ...
    'Tvals', 'Btotal', 'blockSize', 'lambda', 'seed0', 'nMC', 'tagPrefix', ...
    'cov_opts', 'covLabel', 'revision_id', 'resume', 'design_fingerprint');
[okMat, msgMat] = movefile(matTmp, matFile, 'f');
assert(okMat, 'Could not finalize %s: %s', matFile, msgMat);

fprintf('\nCombined CHJ rejection-study outputs:\n');
fprintf('  %s\n', rawFile);
fprintf('  %s\n', summaryFile);
if ~isempty(regularization_bias_all), fprintf('  %s\n', regRawFile); end
if ~isempty(regularization_bias_summary), fprintf('  %s\n', regSummaryFile); end
if ~isempty(recovery_all), fprintf('  %s\n', recoveryRawFile); end
if ~isempty(recovery_summary), fprintf('  %s\n', recoverySummaryFile); end
fprintf('  %s\n', blockLogFile);
fprintf('  %s\n', matFile);

end

function local_validate_checkpoint(saved, Tvals, Bblock, lambda, seed0, nMC, cov_opts, revision_id, regularizationLambdaPath, design_fingerprint)
%% Verify that a completed block belongs to the requested CHJ study design.

required = {'raw','regularization_bias_raw','recovery_raw','regularizationLambdaPath','Tvals', ...
    'B','lambda','seed0','nMC','cov_opts','revision_id','design_fingerprint'};
for j = 1:numel(required)
    assert(isfield(saved, required{j}), ...
        'Checkpoint is missing required revision field %s.', required{j});
end

assert(isequal(saved.Tvals(:).', Tvals(:).'), ...
    'Checkpoint T values do not match the requested block.');
assert(saved.B == Bblock, 'Checkpoint B does not match the requested block.');
assert(isequal(saved.lambda, lambda), 'Checkpoint lambda mode/value does not match.');
assert(saved.seed0 == seed0, 'Checkpoint seed0 does not match the requested block.');
assert(saved.nMC == nMC, 'Checkpoint nMC does not match the requested block.');
assert(strcmp(char(string(saved.revision_id)), char(string(revision_id))), ...
    'Checkpoint revision id does not match the requested code path.');
assert(strcmp(local_covariance_label(saved.cov_opts), local_covariance_label(cov_opts)), ...
    'Checkpoint covariance design does not match the requested block.');
assert(isequal(saved.regularizationLambdaPath(:), regularizationLambdaPath(:)), ...
    'Checkpoint regularization path does not match the requested revision.');
assert_design_fingerprint(saved.design_fingerprint, design_fingerprint);

end

function out = local_cat_nonempty_tables(cells)
%% Concatenate optional checkpoint tables without 0-by-0 conflicts.

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
%% Fill covariance options used by the block runner.

if nargin < 1 || isempty(cov_opts), cov_opts = struct(); end
if ~isfield(cov_opts, 'type') || isempty(cov_opts.type), cov_opts.type = 'hac_bartlett'; end
if ~isfield(cov_opts, 'bandwidth') || isempty(cov_opts.bandwidth), cov_opts.bandwidth = 0; end
if ~isfield(cov_opts, 'bandwidth_rule') || isempty(cov_opts.bandwidth_rule), cov_opts.bandwidth_rule = 'nw_4_t100_2over9'; end
if ~isfield(cov_opts, 'winsor_margin') || isempty(cov_opts.winsor_margin), cov_opts.winsor_margin = 0; end
if ~isfield(cov_opts, 'recenter') || isempty(cov_opts.recenter), cov_opts.recenter = false; end

end

function label = local_covariance_label(cov_opts)
%% Build the covariance label used in checkpoint metadata.

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
