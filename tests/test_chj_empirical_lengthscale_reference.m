function test_chj_empirical_lengthscale_reference()
%% Verify CHJ full-panel lengthscales on the forward-moneyness grid.

cfg = config_chj_simulation_design();
m = chj_option_moneyness_grid(cfg);

% Exact median pairwise distance on three repeated daily-domain option panels.
thetaM = median_pairwise_lengthscale_weighted(m, repmat(3, numel(m), 1));
expandedM = repmat(m, 3, 1);
D = abs(expandedM - expandedM.');
expectedThetaM = median(D(triu(true(numel(expandedM)), 1)));
assert(abs(thetaM - expectedThetaM) <= 1e-14);

% scipy cityblock pdist on four date-level VIX states, each repeated J times.
v = [20; 80; 110; 140];
thetaV = median_pairwise_lengthscale_weighted(v, repmat(cfg.sim.J, 4, 1));
assert(abs(thetaV - 60) <= 1e-12);

fprintf('test_chj_empirical_lengthscale_reference passed\n');

end
