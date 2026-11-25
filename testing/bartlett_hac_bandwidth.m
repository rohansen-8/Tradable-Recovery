function L = bartlett_hac_bandwidth(T)
%% Return the documented Newey--West/Bartlett lag length for sample size T.
%
% The common empirical/simulation rule is
%       L_T = floor(4 * (T / 100)^(2/9)).

assert(isscalar(T) && isfinite(T) && T >= 1 && T == floor(T), ...
    'T must be a positive integer scalar.');

L = floor(4 * (T / 100)^(2/9));

end
