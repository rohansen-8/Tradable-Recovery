function outDir = simulation_output_dir(kind, lambdaMethod)
%% Return the simulation results or figures root for one lambda rule.
%
%   kind         : 'simulations' or 'figures'
%   lambdaMethod : 'cv' (results/<kind>/cv/),
%                  'eff_dim' (results/<kind>/eff_dim/) or
%                  'fixed_lambda' (results/<kind>/fixed_lambda/)
%
% The environment variable SDF_RESULTS_ROOT, when set, replaces
% <project>/results (see results_root).

kind = validatestring(char(string(kind)), {'simulations','figures'}, ...
    mfilename, 'kind');
lambdaMethod = validatestring(char(string(lambdaMethod)), {'cv','eff_dim','fixed_lambda'}, ...
    mfilename, 'lambdaMethod');

outDir = fullfile(results_root(), kind, lambdaMethod);

end
