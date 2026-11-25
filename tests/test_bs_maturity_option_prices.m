function test_bs_maturity_option_prices()
%% Verify BS option pricing on the horizon-specific simulated strike grids.

setpath();

horizons = ["daily"; "weekly"; "monthly"; "quarterly"];
atmPrice = zeros(numel(horizons), 1);

for j = 1:numel(horizons)
    cfg = config_bs_simulation_design(horizons(j),'bs_empirical_t');
mon_grid = linspace(cfg.training.estimator_domain(1), cfg.training.estimator_domain(2), cfg.training.n_m_grid).';
    pre = build_bs_alpha_precompute(cfg, mon_grid);

    % Every stored option price must correspond to its actual observed
    % forward-moneyness location. The OTM option grid need not contain m=1.
    qCall = bs_forward_option_price(pre.mGrid, cfg.bs.sigma, cfg.bs.dt, 'call');
    qPut = bs_forward_option_price(pre.mGrid, cfg.bs.sigma, cfg.bs.dt, 'put');
    qExpected = qPut;
    qExpected(pre.mGrid > 1) = qCall(pre.mGrid > 1);

    assert(max(abs(pre.q - qExpected)) < 1e-14, ...
        'Stored BS option prices do not match their simulated moneyness grid.');

    % m=1 remains the separate first-derivative reference point even when it
    % is not an observed option strike. Compare maturities at this common point.
    atmPrice(j) = bs_forward_option_price(1, cfg.bs.sigma, cfg.bs.dt, 'put');
end

assert(all(diff(atmPrice) > 0), ...
    'ATM BS option price should increase with the retained maturity sequence.');

cfg = config_bs_simulation_design('monthly','bs_empirical_t');
qMonthly = bs_forward_option_price(1, cfg.bs.sigma, 1/12, 'put');
assert(abs(atmPrice(3) - qMonthly) < 1e-14, ...
    'Monthly option pricing must remain the original 1/12-year baseline.');

fprintf('test_bs_maturity_option_prices passed.\n');

end
