function run_bs_tests()
%% Run the unit tests for the Black--Scholes simulation pipeline.

test_cross_validate_lambda_python_reference;
test_cross_validate_lambda_scale_invariance;
test_empirical_maturity_domain_identity;
test_config_bs_maturity_design;
test_bs_study_design;
test_bs_empirical_tuning_rules;
test_bs_maturity_option_prices;
test_build_bs_alpha_from_returns;
test_summarize_bs_rejection_results;
test_bs_power_diagnostics;
test_bs_conv_conc_calibration_identity;

fprintf('run_bs_tests passed.\n');

end
