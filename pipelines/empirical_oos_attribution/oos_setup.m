function sdfRoot = oos_setup(sdfRoot)
%% Put the canonical SDF code and this pipeline on the path, and verify both.
%
% The estimator, configuration and evaluation functions are reused from the
% canonical checkout (default: this repository). This folder is added after it, so no
% pipeline file can shadow a canonical function. The check at the end fails
% if any reused function would resolve outside sdfRoot (for example from a
% stale copy of the repository on the path).

if nargin < 1 || isempty(sdfRoot)
    sdfRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % repository root
end
sdfRoot = char(java.io.File(sdfRoot).getCanonicalPath());
assert(isfile(fullfile(sdfRoot, 'drivers', 'setpath.m')), ...
    'oos_setup:BadRoot', 'No drivers/setpath.m under %s.', sdfRoot);

run(fullfile(sdfRoot, 'drivers', 'setpath.m'));
here = fileparts(mfilename('fullpath'));
addpath(here, '-end');

reused = {'config_empirical','get_frozen_design','get_empirical_maturity_config', ...
    'train_projection_model','build_alpha_panel','build_kernel_norm_matrix', ...
    'build_return_second_moment','build_payoff_factor','select_lambda_effective_dimension', ...
    'solve_projection_qr','option_level_median_lengthscales','save_model_state', ...
    'build_exo_representer','evaluate_sdf_derivatives','build_theta_hat', ...
    'bartlett_hac_bandwidth','verify_saved_fit','get_project_root'};
for j = 1:numel(reused)
    w = which(reused{j});
    assert(startsWith(w, [sdfRoot filesep]), 'oos_setup:Shadowed', ...
        '%s resolves to %s, outside %s.', reused{j}, w, sdfRoot);
end
assert(strcmp(get_project_root(), sdfRoot), 'oos_setup:Root', ...
    'get_project_root() = %s, expected %s.', get_project_root(), sdfRoot);

end
