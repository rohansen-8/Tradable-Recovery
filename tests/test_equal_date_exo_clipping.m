function test_equal_date_exo_clipping()
%% Verify equal-date clipping and weighting against the empirical convention.

cfg = config_chj_simulation_design();
cfg.kernel.exo.lengthscale = 2;

model_state = struct();
model_state.exo_grid = [10; 20; 30];
model_state.ls_exo = 2;

obs = [5; 10; 25; 40];
opts = struct('mode', 'equal_date', 'observations', obs, 'clip_to_grid', true);
exo = build_exo_representer(cfg, model_state, [], opts);

expectedSupport = [10; 10; 25; 30];
expectedWeights = ones(4, 1) / 4;
expectedE = laplace_k(model_state.exo_grid, expectedSupport.', 2) * expectedWeights;

assert(isequal(exo.support, expectedSupport));
assert(isequal(exo.support_raw, obs));
assert(exo.n_clipped == 2);
assert(abs(exo.frac_clipped - 0.5) <= 1e-12);
assert(max(abs(exo.weights - expectedWeights)) <= 1e-12);
assert(max(abs(exo.e - expectedE)) <= 1e-12);

fprintf('test_equal_date_exo_clipping passed\n');

end
