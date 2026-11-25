function run_chj_tests()
%% Run the unit tests for the CHJ simulation pipeline.

setpath();

test_empirical_maturity_domain_identity();
testDir = fileparts(mfilename('fullpath'));
unitResult = runtests(fullfile(testDir, 'test_config_chj_simulation_design.m'));
assert(all([unitResult.Passed]), ...
    'test_config_chj_simulation_design contains a failing unit test.');

test_config_chj_study_design();
test_chj_study_design();
test_chj_empirical_lengthscale_reference();
test_chj_empirical_grid_alignment();
test_median_pairwise_lengthscale_scalable();
test_median_pairwise_lengthscale_weighted();
test_equal_date_exo_clipping();
test_chj_numeric_path_reuse();
test_cross_validate_lambda_prewhiten_equivalence();
test_cross_validate_lambda_python_reference();
test_build_mc_pvalue_geometry_reuse();

fprintf('run_chj_tests passed\n');

end
