function test_empirical_maturity_domain_identity()
%% Verify the fixed per-maturity empirical domains.
%
% Only the empirical maturity constants are checked here. The simulation
% experiments align generation, estimation, and inference on one common
% domain; their grids are verified by verify_simulation_design.

setpath();
horizons = ["quarterly"; "monthly"; "weekly"; "daily"];
expectedSupport = [ ...
    0.56147167769263,   1.18831169511829; ...
    0.744456334784572,  1.09089309666474; ...
    0.917587749971634,  1.03688696897526; ...
    0.967264263145474,  1.0208603191431];
expectedEstimator = [0.7 1.4; 0.7 1.3; 0.7 1.3; 0.8 1.2];
expectedCuts = [0.880,1.030; 0.920,1.030; 0.945,1.000; 0.970,1.010];
expectedFigureLog = [-0.25 0.25; -0.30 0.15; -0.20 0.10; -0.10 0.10];

for j = 1:numel(horizons)
    maturity = get_empirical_maturity_config(horizons(j));
    assert(isequal(maturity.support_domain, expectedSupport(j,:)));
    assert(isequal(maturity.estimator_domain, expectedEstimator(j,:)));
    assert(isequal(maturity.region_cutoffs, expectedCuts(j,:)));
    assert(isequal(maturity.figure_log_range, expectedFigureLog(j,:)));
    expectedTest = [max(exp(expectedFigureLog(j,1)), expectedSupport(j,1)), ...
        min(exp(expectedFigureLog(j,2)), expectedSupport(j,2))];
    assert(isequal(maturity.test_domain, expectedTest));
    assert(maturity.test_domain(1) < maturity.region_cutoffs(1) && ...
        maturity.region_cutoffs(2) < maturity.test_domain(2));
end

% Only quarterly's test domain is figure-limited; the other three equal
% their supports exactly.
q = get_empirical_maturity_config('quarterly');
assert(q.test_domain(1) == exp(-0.25) && q.test_domain(2) == q.support_domain(2));

% The empirical estimator basis and 60-midpoint shape grid are deterministic
% functions of the constants above.
for j = 1:numel(horizons)
    maturity = get_empirical_maturity_config(horizons(j));
    estimatorGrid = linspace(maturity.estimator_domain(1), ...
        maturity.estimator_domain(2), 30).';
    assert(estimatorGrid(1) == maturity.estimator_domain(1));
    assert(estimatorGrid(end) == maturity.estimator_domain(2));
    dm = diff(maturity.test_domain) / 60;
    expectedShapeGrid = maturity.test_domain(1) + ((0:59).' + 0.5) * dm;
    assert(isequal(build_global_shape_grid(config_empirical(horizons(j))), ...
        expectedShapeGrid));
end

fprintf('test_empirical_maturity_domain_identity passed.\n');

end
