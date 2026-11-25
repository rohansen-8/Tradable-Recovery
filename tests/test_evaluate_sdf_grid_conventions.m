function test_evaluate_sdf_grid_conventions()
%% Convention test for h evaluation and the returned M fields.

cfg = struct();
cfg.has_exo = false;

model_state = struct();
model_state.mon_grid = [0.9; 1.0; 1.1];
model_state.reference_point = 1;
model_state.ls_endo = 0.15;
model_state.h_star = [0.1; -0.2; 0.05; 0.03];

m_grid = linspace(0.85, 1.15, 9).';

level = evaluate_sdf_grid(cfg, model_state, m_grid, []);
derivs = evaluate_sdf_derivatives(cfg, model_state, m_grid, []);

assert(isfield(level, 'h') && isfield(level, 'M'), ...
    'evaluate_sdf_grid must return h and M.');

assert(isfield(derivs, 'h_m') && isfield(derivs, 'h_mm'), ...
    'evaluate_sdf_derivatives must return h_m and h_mm.');

assert(isfield(derivs, 'M_m') && isfield(derivs, 'M_mm'), ...
    'evaluate_sdf_derivatives must return M_m and M_mm.');

assert(numel(level.h) == numel(m_grid), 'h has wrong length.');
assert(numel(level.M) == numel(m_grid), 'M has wrong length.');
assert(numel(derivs.h_m) == numel(m_grid), 'h_m has wrong length.');
assert(numel(derivs.M_mm) == numel(m_grid), 'M_mm has wrong length.');

assert(norm(level.M - (1 - level.h), inf) < 1e-12, ...
    'Display transform failed: M must equal 1 - h.');

assert(norm(derivs.M_m + derivs.h_m, inf) < 1e-12, ...
    'First derivative sign convention failed: M_m must equal -h_m.');

assert(norm(derivs.M_mm + derivs.h_mm, inf) < 1e-12, ...
    'Second derivative sign convention failed: M_mm must equal -h_mm.');

fprintf('test_evaluate_sdf_grid_conventions passed.\n');
end
