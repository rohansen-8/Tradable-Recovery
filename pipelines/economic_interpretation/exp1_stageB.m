function exp1_stageB(h, outRoot)
%% Experiment 1, step 2: Ledoit-Wolf Sharpe-ratio difference test, forward leg vs c_t * Delta.
% lw_sharpe_difference_test with its default settings (M 4999, K 5000, Minner 499,
% SB10 calibration, grid {1,2,4,6,8,10} with b <= T/10, fixed seeds). Block sensitivity
% reuses the observed HAC statistic and redraws only the final bootstrap at each b.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % repository root
cd(root); run('drivers/setpath.m');
if nargin < 2 || isempty(outRoot), outRoot = fullfile(results_root(),'economic_interpretation','eff_dim'); end
outDir = fullfile(outRoot,'exp1',h);   % outRoot: default results/economic_interpretation/eff_dim
fB = fullfile(outDir, sprintf('%s_exp1_lw.mat', h));
assert(~isfile(fB), 'exp1:Exists', 'Refusing to overwrite %s', fB);
ann = struct('monthly',12,'weekly',52,'daily',252);
T = readtable(fullfile(outDir, sprintf('%s_exp1_accounting.csv',h)));
r = [T.forward, T.const_payoff];
tic; res = lw_sharpe_difference_test(r, ann.(h)); tTest = toc;
o = res.opts; hs = lw_sharpe_hac_se(r);
blocks = o.block_grid(o.block_grid <= size(r,1)/10); p2 = nan(size(blocks)); p1 = p2;
for j = 1:numel(blocks)
    [ds, ss] = lw_sharpe_boot_draws(r, blocks(j), o.M, RandStream('twister','Seed',o.seed_final));
    ok = isfinite(ds) & isfinite(ss) & ss > 0; ts = (ds - hs.delta) ./ ss;
    p2(j) = (1 + sum(abs(ts(ok)) >= abs(hs.d))) / (sum(ok) + 1);
    p1(j) = (1 + sum(ts(ok) >= hs.d)) / (sum(ok) + 1);
end
res.elapsed_seconds = tTest; res.sensitivity = table(blocks.', p2.', p1.', ...
    'VariableNames', {'block','p_two_sided','p_one_sided_fwd_gt_const'});
save(fB, 'res', '-v7');
writetable(res.sensitivity, fullfile(outDir, sprintf('%s_exp1_lw_block_sensitivity.csv',h)));
row = table(string(h), res.T, res.sr_strategy_annual, res.sr_benchmark_annual, res.delta_sr_annual, ...
    res.se_period*sqrt(ann.(h)), res.d_observed, res.block_size, res.p_value_two_sided, res.n_valid, tTest, ...
    'VariableNames', {'horizon','T','sr_forward','sr_const','delta_sr','se_delta_sr_ann','d','block','p_two_sided','n_valid','seconds'});
writetable(row, fullfile(outDir, sprintf('%s_exp1_lw_summary.csv',h)));
disp(row); disp(res.sensitivity); disp(res.calibration.coverage);
end
