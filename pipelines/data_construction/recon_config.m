function c = recon_config()
%% Paths and cleaning constants for the option-panel reconstruction.
%
% Constants follow the authors' Python data pipeline, OptPortOpt_V2 (data/cleaning/
% cleaning.py, preparing_quarterly.py, preparing_monthlyV2.py with
% ENTRY = 'entry_monday', preparing_weeklyV2.py, preparing_daily.py, and the
% period_dates of config.py). They are recorded in every manifest.

c.root = get_project_root();
c.rec = fullfile(c.root, 'data_reconstructed');
c.raw = fullfile(c.rec, 'raw');
c.extracted = fullfile(c.rec, 'extracted');
c.baseline = fullfile(c.rec, 'baseline_replication');
c.diagnostics = fullfile(c.rec, 'diagnostics');
c.reports = fullfile(c.rec, 'reports');
c.panels_existing = fullfile(c.root, 'data');
c.zip = struct('options', 'wrds_spx_option_prices.zip', 'forwards', 'wrds_spx_forward_prices.zip', ...
    'zerocd', 'wrds_zero_coupon_yield.zip', 'spx', 'wrds_spx_index_prices.zip', 'vix', 'wrds_cboe_vix.zip');
c.horizons = {'quarterly', 'monthly', 'weekly', 'daily'};

% cleaning.py
c.clean_dates = [datetime(2000, 1, 1), datetime(2025, 12, 31)];
c.tol_atm = 0.0015;              % ATM/ATF tolerance on log (forward) moneyness
c.mat_prefilter = [1:9, 25:33, 80:100];   % union of the four maturity windows (dedup key contains mat)

% preparing_* (period_dates 'full'; entry_monday windows)
c.q = struct('dates', [datetime(2000, 1, 1), datetime(2025, 8, 29)], 'months', [3 6 9 12], 'mat', [80 100]);
c.m = struct('dates', [datetime(2000, 1, 1), datetime(2025, 8, 29)], 'months', 1:12, 'mat', [25 33]);
c.w = struct('dates', [datetime(2011, 1, 1), datetime(2025, 8, 29)], 'mat', [5 9], 'target', 7);
c.d = struct('dates', [datetime(2016, 1, 1), datetime(2025, 8, 29)], 'mat', [1 4], 'target', 1, 'min_options', 10);
c.saturday_expiry_before = datetime(2015, 2, 1);   % standard SPX expiration moved from Saturday to Friday
end
