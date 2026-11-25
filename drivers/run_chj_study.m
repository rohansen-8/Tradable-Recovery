function outputs = run_chj_study(Btotal, blockSize, nMC, useParallel, tag, resume, Tfilter, lambdaMethod, fixedLambda)
%% Run the five-sample-size CHJ simulation study.
%
% CV uses the Woodbury solver with periodic refresh. Each T is checkpointed in
% small replication blocks, checked before the next T starts, and saved
% independently so an interrupted run resumes from completed blocks.

setpath();
% lambdaMethod ('cv' or 'eff_dim'; default get_frozen_design().
% simulation_lambda_method) selects the fitting-lambda rule. Outputs go to
% simulation_output_dir('simulations', lambdaMethod)/<tag>. lambdaMethod
% 'fixed_lambda' uses the one supplied fixedLambda for every T and replication
% (no selection) and writes to .../fixed_lambda/<tag>/lambda_<tag>/.
if nargin < 9, fixedLambda = []; end
if nargin < 8 || isempty(lambdaMethod)
    lambdaMethod = get_frozen_design().simulation_lambda_method;
end
lambdaMethod = validatestring(char(string(lambdaMethod)), {'cv','eff_dim','fixed_lambda'}, ...
    mfilename, 'lambdaMethod');
assert(strcmp(lambdaMethod, 'fixed_lambda') || isempty(fixedLambda), ...
    'fixedLambda applies to lambdaMethod ''fixed_lambda'' only.');
cfg = apply_fixed_lambda(config_chj_simulation_design(lambdaMethod), fixedLambda);

if nargin < 1 || isempty(Btotal), Btotal = cfg.sim.final_B; end
if nargin < 2 || isempty(blockSize), blockSize = 12; end
if nargin < 3 || isempty(nMC), nMC = cfg.testing.nMC; end
if nargin < 4 || isempty(useParallel), useParallel = true; end
if nargin < 5 || isempty(tag), tag = 'production'; end
if nargin < 6 || isempty(resume), resume = true; end
if nargin < 7 || isempty(Tfilter), Tfilter = cfg.sim.performance_T; end

Btotal = double(Btotal);
blockSize = double(blockSize);
nMC = double(nMC);
useParallel = logical(useParallel);
tag = char(tag);
resume = logical(resume);
Tfilter = double(Tfilter(:).');

assert(Btotal >= 1 && Btotal == floor(Btotal));
assert(blockSize >= 1 && blockSize == floor(blockSize));
assert(nMC >= 1 && nMC == floor(nMC));

verify_simulation_design();
design = get_chj_study_design();
assert(all(ismember(Tfilter, cfg.sim.performance_T)), ...
    'Tfilter contains a CHJ sample size outside the frozen design.');
design = design(ismember(design.Tval, Tfilter), :);
assert(~isempty(design), 'Requested CHJ T filter is empty.');
design.B = repmat(Btotal, height(design), 1);
design.nMC = repmat(nMC, height(design), 1);

outDir = fullfile(simulation_output_dir('simulations', lambdaMethod), tag);
if strcmp(lambdaMethod, 'fixed_lambda'), outDir = fullfile(outDir, fixed_lambda_tag(fixedLambda)); end
progressDir = fullfile(outDir, 'progress');
if ~isfolder(progressDir), mkdir(progressDir); end
writetable(design, fullfile(outDir, 'chj_study_design.csv'));

fprintf('\nFrozen CHJ study design\n');
disp(design(:, {'design_id','Tval','horizon_days','option_panel_m_points', ...
    'option_panel_m_grid_type','option_panel_m_min','option_panel_m_max', ...
    'n_m_grid','n_exo_grid','test_grid_n_points','test_grid_min','test_grid_max', ...
    'cv_min_train_dates','hac_bandwidth','seed0'}));
fprintf('B per design point: %d\n', Btotal);
fprintf('nMC per shape test: %d\n', nMC);
fprintf('regularization path: %s\n', mat2str(cfg.diagnostics.regularization_lambda_path.'));
fprintf('block size: %d\n', blockSize);
fprintf('lambda method: %s\n', lambdaMethod);
fprintf('CV solver: %s, refresh=%d\n', cfg.training.cv.solver, cfg.training.cv.woodbury_refresh);
fprintf('resume: %d\n\n', resume);

nDesign = height(design);
rawCells = cell(nDesign,1);
summaryCells = cell(nDesign,1);
regCells = cell(nDesign,1);
regSummaryCells = cell(nDesign,1);
recoveryCells = cell(nDesign,1);
recoverySummaryCells = cell(nDesign,1);
logCells = cell(nDesign,1);
auditCells = cell(nDesign,1);

for j = 1:nDesign
    Tval = design.Tval(j);
    cov_opts = empirical_covariance(cfg);
    runTag = sprintf('%s_%02d_T%d', tag, design.design_id(j), Tval);

    fprintf('\n############################################################\n');
    fprintf('CHJ design %d / %d: one-day maturity, T=%d\n', j, nDesign, Tval);
    fprintf('############################################################\n');

    [summary_j, raw_j, log_j, reg_j, recovery_j] = ...
        run_chj_simulated_shape_rejection_blocks( ...
        Tval, Btotal, blockSize, fixedLambda, design.seed0(j), nMC, runTag, ...
        useParallel, cov_opts, resume, lambdaMethod);

    expectRecovery = Tval == cfg.diagnostics.recovery_T;
    audit_j = audit_simulation_block_outputs( ...
        "CHJ", cfg, Tval, Btotal, raw_j, reg_j, recovery_j, expectRecovery);

    raw_j.design_id = repmat(design.design_id(j), height(raw_j), 1);
    raw_j.design_group = repmat(design.design_group(j), height(raw_j), 1);
    raw_j = movevars(raw_j, {'design_id','design_group'}, 'Before', 1);

    summary_j.design_id = repmat(design.design_id(j), height(summary_j), 1);
    summary_j.design_group = repmat(design.design_group(j), height(summary_j), 1);
    summary_j = movevars(summary_j, {'design_id','design_group'}, 'Before', 1);

    if ~isempty(reg_j)
        reg_j.design_id = repmat(design.design_id(j), height(reg_j), 1);
        reg_j.design_group = repmat(design.design_group(j), height(reg_j), 1);
        reg_j = movevars(reg_j, {'design_id','design_group'}, 'Before', 1);
        reg_summary_j = summarize_regularization_sensitivity(reg_j);
    else
        reg_summary_j = table();
    end

    if ~isempty(recovery_j)
        recovery_j.design_id = repmat(design.design_id(j), height(recovery_j), 1);
        recovery_j.design_group = repmat(design.design_group(j), height(recovery_j), 1);
        recovery_j = movevars(recovery_j, {'design_id','design_group'}, 'Before', 1);
        recovery_summary_j = summarize_sdf_recovery(recovery_j);
    else
        recovery_summary_j = table();
    end

    log_j.design_id = repmat(design.design_id(j), height(log_j), 1);
    log_j.design_group = repmat(design.design_group(j), height(log_j), 1);
    log_j.Tval = repmat(Tval, height(log_j), 1);
    log_j = movevars(log_j, {'design_id','design_group','Tval'}, 'Before', 1);

    audit_j.design_id = design.design_id(j);
    audit_j.design_group = design.design_group(j);
    audit_j = movevars(audit_j, {'design_id','design_group'}, 'Before', 1);

    rawCells{j} = raw_j;
    summaryCells{j} = summary_j;
    regCells{j} = reg_j;
    regSummaryCells{j} = reg_summary_j;
    recoveryCells{j} = recovery_j;
    recoverySummaryCells{j} = recovery_summary_j;
    logCells{j} = log_j;
    auditCells{j} = audit_j;

    writetable(audit_j, fullfile(progressDir, sprintf('design_%02d_T%d_audit.csv', ...
        design.design_id(j), Tval)));
    writetable(summary_j, fullfile(progressDir, sprintf('design_%02d_T%d_rejection_summary.csv', ...
        design.design_id(j), Tval)));
    if ~isempty(reg_summary_j)
        writetable(reg_summary_j, fullfile(progressDir, sprintf('design_%02d_T%d_lambda_summary.csv', ...
            design.design_id(j), Tval)));
    end
    if ~isempty(recovery_summary_j)
        writetable(recovery_summary_j, fullfile(progressDir, ...
            sprintf('design_%02d_T%d_recovery_summary.csv', design.design_id(j), Tval)));
    end
end

raw_all = cat(1, rawCells{:});
rejection_summary = cat(1, summaryCells{:});
regularization_bias_raw = local_cat_nonempty_tables(regCells);
regularization_bias_summary = local_cat_nonempty_tables(regSummaryCells);
recovery_raw = local_cat_nonempty_tables(recoveryCells);
recovery_summary = local_cat_nonempty_tables(recoverySummaryCells);
block_log = cat(1, logCells{:});
audit_table = cat(1, auditCells{:});

compactVars = {'design_id','design_group','Tval','horizon_days','B', ...
    'valid_tests','region','testType','role','lambda_median','lambda_p25', ...
    'lambda_p75','lambda_lower_bound_freq','lambda_upper_bound_freq', ...
    'lambda_fit_median','lambda_inference_median','lambda_changed_freq', ...
    'lambda_ratio_median','ls_endo', ...
    'ls_exo_median','ls_exo_p25','ls_exo_p75','exo_frac_clipped_mean', ...
    'covariance_bandwidth','rej_1pct','se_1pct','rej_5pct','se_5pct', ...
    'rej_10pct','se_10pct'};
compact_table = rejection_summary(:, compactVars);
compact_table = sortrows(compact_table, {'design_id','testType','region'});
rejection_summary = sortrows(rejection_summary, {'design_id','testType','region'});
if ~isempty(regularization_bias_raw)
    regularization_bias_raw = sortrows(regularization_bias_raw, ...
        {'design_id','seed','testType','region','lambda'}, ...
        {'ascend','ascend','ascend','ascend','descend'});
end
if ~isempty(regularization_bias_summary)
    regularization_bias_summary = sortrows(regularization_bias_summary, ...
        {'design_id','testType','region','lambda'}, ...
        {'ascend','ascend','ascend','descend'});
end
if ~isempty(recovery_raw), recovery_raw = sortrows(recovery_raw, {'design_id','seed','m'}); end
if ~isempty(recovery_summary), recovery_summary = sortrows(recovery_summary, {'design_id','m'}); end

writetable(compact_table, fullfile(outDir, 'chj_compact_table.csv'));
writetable(rejection_summary, fullfile(outDir, 'chj_rejection_summary.csv'));
writetable(raw_all, fullfile(outDir, 'chj_raw.csv'));
if ~isempty(regularization_bias_raw)
    writetable(regularization_bias_raw, fullfile(outDir, 'chj_regularization_bias_raw.csv'));
    writetable(regularization_bias_summary, fullfile(outDir, 'chj_regularization_bias_summary.csv'));
end
if ~isempty(recovery_raw), writetable(recovery_raw, fullfile(outDir, 'chj_recovery_raw.csv')); end
if ~isempty(recovery_summary), writetable(recovery_summary, fullfile(outDir, 'chj_recovery_summary.csv')); end
writetable(block_log, fullfile(outDir, 'chj_block_log.csv'));
writetable(audit_table, fullfile(outDir, 'chj_audit.csv'));

revision_id = cfg.revision.id;
regularization_lambda_path = cfg.diagnostics.regularization_lambda_path(:);
run_metadata = simulation_run_metadata(lambdaMethod);
if ~isempty(fixedLambda), run_metadata.fixed_lambda = fixedLambda; end
writetable(struct2table(run_metadata), fullfile(outDir, 'chj_run_metadata.csv'));
save(fullfile(outDir, 'chj_results.mat'), ...
    'design','compact_table','rejection_summary','raw_all', ...
    'regularization_bias_raw','regularization_bias_summary', ...
    'regularization_lambda_path','recovery_raw','recovery_summary', ...
    'block_log','audit_table','Btotal','blockSize','nMC','useParallel', ...
    'tag','resume','Tfilter','revision_id','lambdaMethod','fixedLambda','run_metadata','-v7.3');

outputs = struct();
outputs.design = design;
outputs.compact_table = compact_table;
outputs.rejection_summary = rejection_summary;
outputs.regularization_bias_raw = regularization_bias_raw;
outputs.regularization_bias_summary = regularization_bias_summary;
outputs.recovery_raw = recovery_raw;
outputs.recovery_summary = recovery_summary;
outputs.block_log = block_log;
outputs.audit = audit_table;
outputs.output_dir = outDir;

fprintf('\nFinished the CHJ study design.\n');
fprintf('Final outputs: %s\n', outDir);

end

function out = local_cat_nonempty_tables(cells)
%% Concatenate optional design tables without 0-by-0 conflicts.

keep = ~cellfun(@isempty, cells);
if any(keep), out = cat(1, cells{keep}); else, out = table(); end

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
