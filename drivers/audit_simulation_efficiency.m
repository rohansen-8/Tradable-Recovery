function audit = audit_simulation_efficiency(lambdaMethod)
%% Check and tabulate the cost of the simulation studies before a production run.
%
% lambdaMethod: 'cv' (default) or 'eff_dim'. The CV solver settings are
% asserted in both cases because they define the CV robustness results.

setpath();
if nargin < 1 || isempty(lambdaMethod), lambdaMethod = 'cv'; end
lambdaMethod = validatestring(char(string(lambdaMethod)), {'cv','eff_dim'}, ...
    mfilename, 'lambdaMethod');
bs = config_bs_simulation_design('monthly','bs_increasing_t');
bsEmp = config_bs_simulation_design('quarterly','bs_empirical_t');
chj = config_chj_simulation_design();
bsDesign = get_bs_study_design();
chjDesign = get_chj_study_design();

assert(strcmp(bs.training.cv.solver, 'fold_eig'));
assert(strcmp(chj.training.cv.solver, 'fold_eig_woodbury'));
assert(chj.training.cv.woodbury_refresh == 40);
assert(~bs.diagnostics.run_per_rep_regularization_path);
assert(~chj.diagnostics.run_per_rep_regularization_path);
assert(strcmp(bs.testing.inference_lambda.rule, 'lambda_fit'));
assert(strcmp(chj.testing.inference_lambda.rule, 'lambda_fit'));

stage = ["BS increasing T"; "BS empirical T"; "CHJ increasing T"];
design_points = [sum(bsDesign.design_group=="monthly_performance"); ...
    sum(bsDesign.design_group=="empirical_T"); height(chjDesign)];
B_per_design = [bs.sim.final_B; bsEmp.sim.final_B; chj.sim.final_B];
production_datasets = design_points .* B_per_design;
block_size = [25; 25; 12];
checkpoint_blocks = design_points .* ceil(B_per_design ./ block_size);
if strcmp(lambdaMethod, 'eff_dim')
    % One generalized eigendecomposition of (Q_alpha, K_h) per replication
    % (31x31 BS, 930x930 CHJ) and a scalar root in log lambda; no CV folds.
    cv_solver = repmat("eff_dim (no CV)", 3, 1);
else
    cv_solver = [string(bs.training.cv.solver); string(bs.training.cv.solver); ...
        string(chj.training.cv.solver)];
end
inference_extra_solve_max = zeros(3,1);
per_rep_regularization_extra_solves = zeros(3,1);
recovery_refits = zeros(3,1);
parallel_workers = repmat(12,3,1);

memory_note = [ ...
    "31-dim BS basis; monthly fixed kernel objects reused within blocks"; ...
    "31-dim BS basis; maturity-matched strike objects reused within blocks"; ...
    "930-dim CHJ basis; one VIX path/grid per replication"];
resume_note = [ ...
    "checkpoint every 25 replications per design"; ...
    "checkpoint every 25 replications per design"; ...
    "checkpoint every 12 replications per T"];
mc_calibrations_per_dataset = repmat(6, 3, 1);   % mono + conv per region;
                                                 % conc reuses conv's draws
shape_tests_per_dataset = repmat(9, 3, 1);       % 3 regions x 3 restrictions
nMC_draws = repmat(10000, 3, 1);

audit = table(stage, design_points, B_per_design, production_datasets, ...
    block_size, checkpoint_blocks, cv_solver, inference_extra_solve_max, ...
    per_rep_regularization_extra_solves, recovery_refits, ...
    shape_tests_per_dataset, mc_calibrations_per_dataset, nMC_draws, ...
    parallel_workers, memory_note, resume_note);

fprintf('\nFINAL SIMULATION EFFICIENCY AUDIT\n');
fprintf('lambda_inference = lambda_fit uses the fitted state directly: zero extra CV fits.\n');
fprintf('Shape inference requires zero additional regularized solves.\n');
fprintf('Per-replication regularization-sensitivity solves are disabled in production.\n');
fprintf('Population-bias diagnostics remain separate from the three production blocks.\n');
fprintf('Recovery is taken from the production fit: zero duplicate simulation/CV/refits.\n');
fprintf('Parallelism is only across independent replications, using 12 process workers.\n');
fprintf('Fitting lambda rule: %s.\n', lambdaMethod);
if strcmp(lambdaMethod, 'eff_dim')
    fprintf(['eff_dim, per replication: one thin QR of the T-by-(Nh+1) payoff panel, ' ...
        'one SVD of the Nh-by-Nh whitened factor for lambda, one QR of the stacked ' ...
        '2Nh-by-(Nh+1) system for h; the same triangular factor serves the score/HAC ' ...
        'inverse actions. CV is not run.\n']);
else
    fprintf('CHJ retains fold_eig_woodbury with refresh = 40.\n');
end
fprintf('Completed blocks are restartable and reused with resume=true.\n\n');
fprintf('Every resume is guarded by the complete final design fingerprint.\n');
fprintf(['B is the number of independent simulated datasets; nMC is the ' ...
    'Gaussian null calibration inside one shape test.\n\n']);
disp(audit);

end
