function test_config_bs_maturity_design()
%% Verify the BS maturity-matched configurations and their three domains.

setpath();
cfgA=config_bs_simulation_design('monthly','bs_increasing_t');
assert(cfgA.sim.J==30);
assert(cfgA.sim.final_B==1000);
% All three simulation supports coincide on the fixed increasing-T domain.
assert(isequal([cfgA.sim.m_min cfgA.sim.m_max],[0.95 1.05]));
assert(isequal(cfgA.training.estimator_domain,[0.95 1.05]));
assert(isequal(cfgA.testing.grid.domain,[0.95 1.05]));
assert(isequal(vertcat(cfgA.testing.regionSpecs.bounds), ...
    [0.95 0.985; 0.985 1.015; 1.015 1.05]));
assert(isequal(cfgA.sim.option_m_grid(:),linspace(0.95,1.05,30).'));
assert(cfgA.training.n_m_grid==30 && cfgA.testing.grid.n_points==52);
% 15 puts and 15 calls: no contract sits on the ambiguous ATM point.
assert(~any(abs(cfgA.sim.option_m_grid-1)<=1e-12));
assert(sum(cfgA.sim.option_m_grid<1)==15 && sum(cfgA.sim.option_m_grid>1)==15);

expectedJ=[46;77;27;28];
final=get_frozen_design();
for j=1:numel(final.bs_empirical_t.horizons)
    horizon=final.bs_empirical_t.horizons(j);
    maturity=get_empirical_maturity_config(horizon);
    domain=exp(maturity.figure_log_range);
    cfg=config_bs_simulation_design(horizon,'bs_empirical_t');
    assert(cfg.sim.J==expectedJ(j));
    assert(cfg.sim.final_B==2000);
    assert(isequal([cfg.sim.m_min cfg.sim.m_max],domain));
    assert(isequal(cfg.training.estimator_domain,domain));
    assert(isequal(cfg.testing.grid.domain,domain));
    % Region cutoffs equal the empirical ones and lie inside the domain.
    assert(isequal(vertcat(cfg.testing.regionSpecs.bounds), ...
        [domain(1) maturity.region_cutoffs(1); ...
         maturity.region_cutoffs(1) maturity.region_cutoffs(2); ...
         maturity.region_cutoffs(2) domain(2)]));
    assert(all(diff(cfg.sim.option_m_grid)>0));
    assert(isequal(cfg.sim.option_m_grid(:), ...
        linspace(domain(1),domain(2),expectedJ(j)).'));
    assert(cfg.training.n_m_grid==30 && cfg.testing.grid.n_points==60);  % empirical-T keeps G=60
    assert(strcmp(cfg.testing.inference_lambda.rule,'lambda_fit'));
end

end
