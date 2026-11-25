function test_chj_study_design()
%% Verify the five-point CHJ sample-size design and daily support.

design = get_chj_study_design();
% Generation, estimator, and test supports all coincide on the fixed
% increasing-T domain.
genDomain = [0.95 1.05];
testDomain = [0.95 1.05];

assert(height(design) == 5);
assert(isequal(design.Tval, [500; 1000; 2000; 5000; 10000]));
assert(all(design.horizon_days == 1));
assert(all(design.option_panel_m_grid_type == "forward_moneyness_linspace"));
assert(all(design.mon_grid_type == "forward_moneyness_linspace"));
assert(all(design.lambda_mode == "cv"));
assert(all(design.lambda_grid_n == 31));
assert(all(abs(design.lambda_grid_min - 1e-5) <= 1e-12));
assert(all(abs(design.lambda_grid_max - 1e-2) <= 1e-12));
assert(all(design.option_panel_m_min == genDomain(1)));
assert(all(design.option_panel_m_max == genDomain(2)));
assert(all(design.cv_min_train_dates == 10));
assert(all(design.kernel_endo_lengthscale_rule == "median"));
assert(all(design.kernel_exo_lengthscale_rule == "median"));
assert(all(design.exo_average_mode == "equal_date"));
assert(all(design.exo_clip_to_grid));
assert(all(design.test_grid_type == "global_midpoints"));
assert(all(design.test_grid_n_points == 52));
assert(all(design.test_grid_min == testDomain(1)));
assert(all(design.test_grid_max == testDomain(2)));
assert(all(design.mon_grid_min == genDomain(1)));
assert(all(design.mon_grid_max == genDomain(2)));
assert(isequal(design.hac_bandwidth, [5; 6; 7; 9; 11]));
assert(all(design.inference_lambda_rule == "lambda_fit"));
assert(all(design.revision_id == "public_v1_chj_increasing_t"));

cfg = config_chj_simulation_design();
option_m_grid = chj_option_moneyness_grid(cfg);
expectedLs = NaN(height(design), 1);
for j = 1:height(design)
    expectedLs(j) = median_pairwise_lengthscale_weighted( ...
        option_m_grid, repmat(design.Tval(j), numel(option_m_grid), 1));
end
assert(all(abs(design.kernel_endo_lengthscale - expectedLs) <= 1e-12));

fprintf('test_chj_study_design passed\n');

end
