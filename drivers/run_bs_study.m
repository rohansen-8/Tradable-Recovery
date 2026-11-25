function outputs = run_bs_study(Btotal, blockSize, nMC, useParallel, tag, resume, designGroup, lambdaMethod, horizons, fixedLambda)
%% Run the nine-point Black--Scholes (BS) simulation study.
%
% lambdaMethod ('cv' or 'eff_dim'; default get_frozen_design().
% simulation_lambda_method) selects the fitting-lambda rule. Outputs go to
% simulation_output_dir('simulations', lambdaMethod)/<tag>. lambdaMethod
% 'fixed_lambda' uses the one supplied fixedLambda for every design point and
% replication (no selection) and writes to .../fixed_lambda/<tag>/lambda_<tag>/.
%
% horizons (optional; empirical_T only) restricts the run to the listed
% maturities. Seeds and design ids come from the full manifest, so a
% restricted run is identical to the corresponding rows of a full run. A
% restricted run writes its top-level files with a _<horizon...> suffix
% (e.g. bs_results_quarterly.mat) and never overwrites another maturity's
% files; per-design block checkpoints are shared and reused either way.
%
% The design comprises the monthly T progression (increasing-T design) and
% the four empirical-scale maturity cases (empirical-T design). Each design
% point is checkpointed by replication block, checked before the next design
% starts, and saved independently so an
% interruption never requires repeating completed work.

setpath();

if nargin < 8 || isempty(lambdaMethod)
    lambdaMethod = get_frozen_design().simulation_lambda_method;
end
lambdaMethod = validatestring(char(string(lambdaMethod)), {'cv','eff_dim','fixed_lambda'}, ...
    mfilename, 'lambdaMethod');
if nargin < 10, fixedLambda = []; end
assert(strcmp(lambdaMethod, 'fixed_lambda') || isempty(fixedLambda), ...
    'fixedLambda applies to lambdaMethod ''fixed_lambda'' only.');
cfgMonthly = apply_fixed_lambda(config_bs_simulation_design('monthly','bs_increasing_t', lambdaMethod), fixedLambda);
if nargin < 1 || isempty(Btotal), Btotal = cfgMonthly.sim.final_B; end
if nargin < 2 || isempty(blockSize), blockSize = 25; end
if nargin < 3 || isempty(nMC), nMC = cfgMonthly.testing.nMC; end
if nargin < 4 || isempty(useParallel), useParallel = true; end
if nargin < 5 || isempty(tag), tag = 'production'; end
if nargin < 6 || isempty(resume), resume = true; end
if nargin < 7 || isempty(designGroup), designGroup = 'all'; end
if nargin < 9, horizons = []; end

Btotal = double(Btotal);
blockSize = double(blockSize);
nMC = double(nMC);
useParallel = logical(useParallel);
tag = char(tag);
resume = logical(resume);
designGroup = string(designGroup);

assert(Btotal >= 1 && Btotal == floor(Btotal));
assert(blockSize >= 1 && blockSize == floor(blockSize));
assert(nMC >= 1 && nMC == floor(nMC));

verify_simulation_design();
design = get_bs_study_design();
if designGroup ~= "all"
    assert(ismember(designGroup, ["empirical_T","monthly_performance"]), ...
        'designGroup must be all, empirical_T, or monthly_performance.');
    design = design(string(design.design_group) == designGroup, :);
end
fileSuffix = '';
if ~isempty(horizons)
    horizons = string(horizons(:).');
    assert(designGroup == "empirical_T", ...
        'A horizon restriction applies to the empirical_T design group only.');
    assert(all(ismember(horizons, string(design.horizon))) && numel(unique(horizons)) == numel(horizons), ...
        'horizons must be distinct empirical_T maturities: %s.', strjoin(string(design.horizon).', ', '));
    design = design(ismember(string(design.horizon), horizons), :);
    fileSuffix = ['_' char(strjoin(string(design.horizon).', '_'))];
end
assert(~isempty(design), 'Requested BS design group is empty.');
design.B = repmat(Btotal, height(design), 1);
design.nMC = repmat(nMC, height(design), 1);

outDir = fullfile(simulation_output_dir('simulations', lambdaMethod), tag);
if strcmp(lambdaMethod, 'fixed_lambda'), outDir = fullfile(outDir, fixed_lambda_tag(fixedLambda)); end
progressDir = fullfile(outDir, 'progress');
if ~isfolder(progressDir), mkdir(progressDir); end
writetable(design, fullfile(outDir, sprintf('bs_study_design%s.csv', fileSuffix)));

fprintf('\nFrozen structural BS study design\n');
disp(design(:, get_bs_display_columns()));
fprintf('B per design point: %d\n', Btotal);
fprintf('nMC per shape test: %d\n', nMC);
fprintf('block size: %d\n', blockSize);
fprintf('lambda method: %s\n', lambdaMethod);
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
    horizon = char(design.horizon(j));
    experiment = char(design.experiment(j));
    Tval = design.Tval(j);
    cfg = apply_fixed_lambda(config_bs_simulation_design(horizon,experiment,lambdaMethod), fixedLambda);
    cov_opts = empirical_covariance(cfg);
    runTag = sprintf('%s_%02d_%s_T%d', tag, design.design_id(j), horizon, Tval);

    fprintf('\n############################################################\n');
    fprintf('BS design %d / %d: %s, T=%d\n', j, nDesign, horizon, Tval);
    fprintf('############################################################\n');

    [summary_j, raw_j, log_j, reg_j, recovery_j] = ...
        run_bs_simulated_shape_rejection_blocks( ...
        Tval, Btotal, blockSize, fixedLambda, design.seed0(j), nMC, runTag, ...
        useParallel, cov_opts, horizon, resume, experiment, lambdaMethod);

    expectRecovery = string(design.design_group(j)) == "monthly_performance" ...
        && Tval == cfg.diagnostics.recovery_T;
    audit_j = audit_simulation_block_outputs( ...
        "BS", cfg, Tval, Btotal, raw_j, reg_j, recovery_j, expectRecovery);

    raw_j.design_id = repmat(design.design_id(j), height(raw_j), 1);
    raw_j.design_group = repmat(design.design_group(j), height(raw_j), 1);
    raw_j.experiment = repmat(design.experiment(j), height(raw_j), 1);
    raw_j = movevars(raw_j, {'design_id','design_group','experiment'}, 'Before', 1);

    summary_j.design_id = repmat(design.design_id(j), height(summary_j), 1);
    summary_j.design_group = repmat(design.design_group(j), height(summary_j), 1);
    summary_j.experiment = repmat(design.experiment(j),height(summary_j),1);
    summary_j = movevars(summary_j, {'design_id','design_group','experiment'}, 'Before', 1);

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
    log_j.horizon = repmat(design.horizon(j), height(log_j), 1);
    log_j.Tval = repmat(Tval, height(log_j), 1);
    log_j = movevars(log_j, {'design_id','design_group','horizon','Tval'}, 'Before', 1);

    audit_j.design_id = design.design_id(j);
    audit_j.design_group = design.design_group(j);
    audit_j.horizon = design.horizon(j);
    audit_j = movevars(audit_j, {'design_id','design_group','horizon'}, 'Before', 1);

    rawCells{j} = raw_j;
    summaryCells{j} = summary_j;
    regCells{j} = reg_j;
    regSummaryCells{j} = reg_summary_j;
    recoveryCells{j} = recovery_j;
    recoverySummaryCells{j} = recovery_summary_j;
    logCells{j} = log_j;
    auditCells{j} = audit_j;

    % Small progress files are written after each fully checked design point.
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

%% Attach known BS population truth and regularization geometry.
% build_bs_power_diagnostics solves one population problem per distinct
% selected lambda. Under CV that is at most 31 grid values; under eff_dim
% every replication has its own continuous lambda (thousands of serial
% population solves), so this reporting-only diagnostic is not run inline.
% It is a deterministic function of the saved raw lambda_fit values and can
% be computed afterwards without touching the simulations.
joinVars = {'design_id','design_group','experiment','horizon','Tval','maturity_years', ...
    'cv_min_train_dates', ...
    'hac_bandwidth_rule','hac_bandwidth','inference_lambda_rule'};
if strcmp(lambdaMethod, 'cv')
    [compact_table, population_diagnostics] = build_bs_power_diagnostics(raw_all);
    compact_table = innerjoin(design(:, joinVars), compact_table, ...
        'Keys', {'design_id','design_group','experiment','horizon','Tval'});
    compact_table = sortrows(compact_table, {'design_id','testType','region'});
else
    compact_table = table();
    population_diagnostics = table();
end
rejection_summary = innerjoin(design(:, joinVars), rejection_summary, ...
    'Keys', {'design_id','design_group','experiment','horizon','Tval'});
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

%% Save the complete final numerical package.
if ~isempty(compact_table), writetable(compact_table, fullfile(outDir, sprintf('bs_compact_table%s.csv', fileSuffix))); end
writetable(rejection_summary, fullfile(outDir, sprintf('bs_rejection_summary%s.csv', fileSuffix)));
if ~isempty(population_diagnostics)
    writetable(population_diagnostics, fullfile(outDir, sprintf('bs_population_diagnostics%s.csv', fileSuffix)));
end
writetable(raw_all, fullfile(outDir, sprintf('bs_raw%s.csv', fileSuffix)));
if ~isempty(regularization_bias_raw)
    writetable(regularization_bias_raw, fullfile(outDir, sprintf('bs_regularization_bias_raw%s.csv', fileSuffix)));
    writetable(regularization_bias_summary, fullfile(outDir, sprintf('bs_regularization_bias_summary%s.csv', fileSuffix)));
end
if ~isempty(recovery_raw), writetable(recovery_raw, fullfile(outDir, sprintf('bs_recovery_raw%s.csv', fileSuffix))); end
if ~isempty(recovery_summary), writetable(recovery_summary, fullfile(outDir, sprintf('bs_recovery_summary%s.csv', fileSuffix))); end
writetable(block_log, fullfile(outDir, sprintf('bs_block_log%s.csv', fileSuffix)));
writetable(audit_table, fullfile(outDir, sprintf('bs_audit%s.csv', fileSuffix)));

revision_id = cfgMonthly.revision.id;
regularization_lambda_path = cfgMonthly.diagnostics.regularization_lambda_path(:);
run_metadata = simulation_run_metadata(lambdaMethod);
if ~isempty(fixedLambda), run_metadata.fixed_lambda = fixedLambda; end
writetable(struct2table(run_metadata), fullfile(outDir, sprintf('bs_run_metadata%s.csv', fileSuffix)));
resultFile = fullfile(outDir, sprintf('bs_results%s.mat', fileSuffix));
save(resultFile, ...
    'design','compact_table','rejection_summary','population_diagnostics', ...
    'raw_all','regularization_bias_raw','regularization_bias_summary', ...
    'regularization_lambda_path','recovery_raw','recovery_summary', ...
    'block_log','audit_table','Btotal','blockSize','nMC','useParallel', ...
    'tag','resume','designGroup','horizons','revision_id','lambdaMethod','fixedLambda', ...
    'run_metadata','-v7.3');

outputs = struct();
outputs.design = design;
outputs.compact_table = compact_table;
outputs.rejection_summary = rejection_summary;
outputs.population_diagnostics = population_diagnostics;
outputs.regularization_bias_raw = regularization_bias_raw;
outputs.regularization_bias_summary = regularization_bias_summary;
outputs.recovery_raw = recovery_raw;
outputs.recovery_summary = recovery_summary;
outputs.block_log = block_log;
outputs.audit = audit_table;
outputs.output_dir = outDir;
outputs.result_file = resultFile;

fprintf('\nFinished the structural BS study design.\n');
fprintf('Final outputs: %s\n', resultFile);

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
