function strike = build_bs_empirical_strike_design(horizon, domain)
%% Build the representative bs_empirical_t strike vector from empirical rows.
%
% The empirical panel supplies one number only: the typical (median across
% dates) count of observed strikes inside this maturity's simulation domain.
% The simulated locations themselves are deterministic and equally spaced
% across the whole domain, so the controlled experiment never inherits the
% empirical strike clustering.
%
% Lengthscale class on the empirical-T grids (see README, "BS lengthscale
% rule on the empirical-T grids"): the BS simulator's grid-only median
% pairwise distance and the option_level_median rule select the same
% pair-distance class at J = 77 (monthly) and J = 27 (weekly), but classes
% one grid step apart at J = 46 (quarterly, 7.69%) and J = 28 (daily, 12.5%).
% The increasing-T studies stay in one class by construction at their strike
% count, which verify_simulation_design asserts.

horizon = validatestring(char(string(horizon)), ...
    {'quarterly','monthly','weekly','daily'}, mfilename, 'horizon');
domain = double(domain(:).');
assert(numel(domain) == 2 && all(isfinite(domain)) && domain(1) < domain(2), ...
    'domain must contain two increasing finite endpoints.');

sourceFile = fullfile(get_project_root(), 'data', horizon, 'df.csv');
assert(isfile(sourceFile), 'Missing empirical panel: %s', sourceFile);
df = readtable(sourceFile, 'VariableNamingRule', 'preserve');
required = {'date','frwd_moneyness'};
assert(all(ismember(required, df.Properties.VariableNames)), ...
    '%s is missing date or frwd_moneyness.', sourceFile);

m = double(df.frwd_moneyness);
assert(all(isfinite(m) & m > 0), 'Empirical forward moneyness must be finite and positive.');
[dateGroup, dates] = findgroups(df.date);
inside = m >= domain(1) & m <= domain(2);
counts = splitapply(@sum, double(inside), dateGroup);
assert(numel(counts) == numel(dates), 'Date-level strike counts are incomplete.');

medianCount = median(counts);
J = round(medianCount);
assert(J >= 2, 'Representative strike count must be at least two.');
pooled = m(inside);
optionGrid = linspace(domain(1), domain(2), J).';
assert(numel(unique(optionGrid)) == J && all(diff(optionGrid) > 0), ...
    'Equally spaced strikes must be strictly increasing and unique.');
assert(optionGrid(1) == domain(1) && optionGrid(end) == domain(2), ...
    'Equally spaced strikes must span the whole simulation domain.');
assert(~any(abs(optionGrid - 1) <= 1e-12), ...
    ['%s equally spaced strikes place a contract exactly at m = 1, where ' ...
     'the OTM contract side is ambiguous.'], horizon);

strike = struct();
strike.horizon = horizon;
strike.source_file = sourceFile;
strike.filter = 'all estimation-panel rows, restricted only to the simulation domain';
strike.domain = domain;
strike.n_dates = numel(dates);
strike.median_count_before_rounding = medianCount;
strike.J = J;
strike.count_quantiles = quantile(counts, [0 0.25 0.50 0.75 1]);
strike.n_pooled = numel(pooled);
strike.moneyness_quantiles = quantile(pooled, [0 0.10 0.25 0.50 0.75 0.90 1]);
strike.probabilities = [];
strike.option_grid = optionGrid;
strike.placement_rule = 'forward_moneyness_linspace_on_simulation_domain';
strike.count_rule = 'median_date_strike_count_inside_simulation_domain';

end
