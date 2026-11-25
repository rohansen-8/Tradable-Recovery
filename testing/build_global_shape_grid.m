function m = build_global_shape_grid(cfg)
%% Build the deterministic common global shape-test grid.
%
% Production simulations use midpoint evaluation so the fitted support and
% inference domain coincide while derivative evaluation stays strictly inside
% the two support endpoints.

assert(isfield(cfg, 'testing') && isfield(cfg.testing, 'grid'), ...
    'Missing cfg.testing.grid.');
assert(strcmpi(cfg.testing.grid.type, 'global_midpoints'), ...
    'Final simulations require cfg.testing.grid.type = global_midpoints.');

domain = cfg.testing.grid.domain(:).';
n = cfg.testing.grid.n_points;
assert(numel(domain) == 2 && domain(1) < domain(2), ...
    'Shape-test domain must be [m_min,m_max].');
assert(n >= 1 && n == floor(n), 'Shape-test n_points must be a positive integer.');

dm = (domain(2) - domain(1)) / n;
m = domain(1) + ((0:n-1).' + 0.5) * dm;

assert(numel(m) == n && m(1) > domain(1) && m(end) < domain(2), ...
    'Global midpoint construction failed.');

end
