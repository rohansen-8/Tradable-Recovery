function pre = build_bs_alpha_precompute(cfg, mon_grid)
%% Precompute seed-invariant matrices for BS alpha construction.
%
% The median-pairwise-distance rule is applied to the configured option
% locations, independently of the 30-point restricted estimator basis.

mon_grid = mon_grid(:);

assert(isfield(cfg.sim, 'option_m_grid') && ~isempty(cfg.sim.option_m_grid), ...
    'The final BS design requires an explicit option_m_grid.');
mGrid = double(cfg.sim.option_m_grid(:));
assert(numel(mGrid) == cfg.sim.J && all(diff(mGrid) > 0), ...
    'The configured BS option grid must contain J strictly increasing points.');
w = local_trapz_weights(mGrid);

if isempty(cfg.kernel.endo.lengthscale)
    assert(isfield(cfg.kernel.endo, 'lengthscale_rule') ...
        && strcmp(cfg.kernel.endo.lengthscale_rule, 'median'), ...
        'BS revision requires the median pairwise-distance lengthscale rule.');
    % option_level_median: the median is taken over option-level rows, that
    % is every strike repeated on each of the T dates, not over the J strike
    % locations alone. The balanced simulated panel has multiplicity T at
    % every strike, so the weighted routine evaluates the documented rule
    % exactly without materializing the J*T-row vector. This is the same
    % call the CHJ study makes on its own option grid.
    ell_m = median_pairwise_lengthscale_weighted(mGrid, ...
        repmat(double(cfg.sim.T), numel(mGrid), 1));
else
    ell_m = cfg.kernel.endo.lengthscale;
end

zbar = cfg.reference_point;

qCall = bs_forward_option_price(mGrid, cfg.bs.sigma, cfg.bs.dt, 'call');
qPut = bs_forward_option_price(mGrid, cfg.bs.sigma, cfg.bs.dt, 'put');

isCall = mGrid > 1;
q = qCall .* isCall + qPut .* (~isCall);

pre = struct();
pre.mGrid = mGrid;
pre.w = w;
pre.q = q;
pre.mon_grid = mon_grid;
pre.n = numel(mon_grid);
pre.ell_m = ell_m;
pre.zbar = zbar;

pre.k00 = matern72_dxy(zbar, zbar, ell_m);
pre.k0m = matern72_dxyy(zbar, mGrid.', ell_m);
pre.g0 = matern72_dxyy(zbar, mon_grid, ell_m);
pre.Kmm = matern72_dxxyy(mon_grid, mGrid.', ell_m);

end

function w = local_trapz_weights(x)
%% Return trapezoidal integration weights on the fixed option grid.

n = numel(x);
w = zeros(n, 1);

w(1) = 0.5 * (x(2) - x(1));
w(n) = 0.5 * (x(n) - x(n - 1));

if n > 2
    w(2:n-1) = 0.5 * (x(3:n) - x(1:n-2));
end

end
