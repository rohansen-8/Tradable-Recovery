function panel = load_oos_panel(oos)
%% Read the estimation panel once and build the per-date table for the OOS run.
%
% The panel is read exactly as the production pipeline reads it
% (readtable(cfg.dataset.dfFile, 'VariableNamingRule', 'preserve')).
%
% Payoff timing: sp500_shifted is the
% SP500 close at the next collection date (quarterly/monthly) or at option
% expiry (weekly/daily, OptPortOpt_V2 get_sp500_at_expiry: the expiry date,
% Saturday/Sunday -> Friday, else the first panel date 1-3 days later, else
% 1-2 days earlier). Option payoffs are intrinsic values at that level and
% SP500_ret = sp500_shifted/SP500 - 1. Each date's realization_date is the
% date whose close equals sp500_shifted, checked against the panel's own
% closes in this order (realization_rule):
%   'next_formation_close' the next panel date;
%   'expiry_close'         the effective expiry date;
%   'expiry_lookup_close'  a panel date in [expiry-2, expiry+3] days;
%   'formation_close_fallback' the formation date itself (the backward
%                          fallback reached it: zero index return, 2 daily
%                          dates in 2016; source data kept as is);
%   'unverified_bound'     no matching close in the panel (the level comes
%                          from a date outside it): expiry + 3 days, the
%                          latest date the lookup can reach, so the embargo
%                          is conservative.
% The payoff is observable at the formation close of any date >= realization_date.
%
% Benchmark conventions (used only for market comparisons, never in the fit):
%   holding_days = realization_date - date (calendar days);
%   rf_hold      = exp(r * holding_days) - 1, r = median_j(-log(bond_j)/mat_j);
%   carry_hold   = exp(c * holding_days) - 1, c = median_j(log(F_j/S)/mat_j),
% i.e. the financing rate of the option premia and the implied forward carry
% (r - q), rescaled from each option's maturity to the actual holding period.
%
% Dates with fewer than protocol.min_options contracts are dropped AFTER the
% timing columns are built on the unfiltered panel.

cfg = oos.cfg;
df = readtable(cfg.dataset.dfFile, 'VariableNamingRule', 'preserve');
need = {'date','frwd_moneyness','OPT_NET_R_PAYOFF','weight','SP500_ret','VIX', ...
    'SP500','sp500_shifted','bond','wrds_forward','mat','flag','bid','ask'};
missing = setdiff(need, df.Properties.VariableNames);
assert(isempty(missing), 'load_oos_panel:Columns', 'Missing columns: %s.', strjoin(missing, ', '));
assert(~any(ismember({'Delta','R'}, df.Properties.VariableNames)), 'load_oos_panel:Delta', ...
    'build_alpha_panel would read Delta/R instead of SP500_ret.');
if ~isdatetime(df.date), df.date = datetime(string(df.date), 'InputFormat', 'yyyy-MM-dd'); end
df = sortrows(df, {'date','frwd_moneyness'});
assert(all(isfinite(df.OPT_NET_R_PAYOFF)) && all(isfinite(df.weight)) && ...
    all(isfinite(df.frwd_moneyness)) && all(isfinite(df.VIX)), 'load_oos_panel:NaN', ...
    'Nonfinite values in estimation columns.');

[g, dates] = findgroups(df.date);
T0 = numel(dates);
D = table(dates, 'VariableNames', {'date'});
D.J = accumarray(g, 1);
D.vix = one_per_date(df.VIX, g, 'VIX');
D.S = one_per_date(df.SP500, g, 'SP500');
D.S_next = one_per_date(df.sp500_shifted, g, 'sp500_shifted');
D.Delta = one_per_date(df.SP500_ret, g, 'SP500_ret');
assert(max(abs(D.Delta - (D.S_next ./ D.S - 1))) <= 1e-12, 'load_oos_panel:Ret', ...
    'SP500_ret differs from sp500_shifted/SP500 - 1.');
D.mat_max = accumarray(g, df.mat, [], @max);
D.n_expiries = accumarray(g, df.mat, [], @(x) numel(unique(x)));

% Realization date of each date's payoff (see header).
expiry = D.date + days(D.mat_max);
wd = weekday(expiry);
expiry(wd == 7) = expiry(wd == 7) - days(1);
expiry(wd == 1) = expiry(wd == 1) - days(2);
D.expiry = expiry;
D.realization_date = expiry + days(3);
D.realization_rule = repmat("unverified_bound", T0, 1);
same = @(i, j) abs(D.S_next(i) - D.S(j)) <= 1e-8 * D.S(j);
for i = 1:T0
    if i < T0 && same(i, i + 1)
        D.realization_date(i) = D.date(i + 1);
        D.realization_rule(i) = "next_formation_close";
        continue
    end
    j = find(D.date == expiry(i), 1);
    if ~isempty(j) && same(i, j)
        D.realization_date(i) = D.date(j);
        D.realization_rule(i) = "expiry_close";
        continue
    end
    for off = [1 2 3 -1 -2]                          % lookup order of the cleaning script
        j = find(D.date == expiry(i) + days(off), 1);
        if ~isempty(j) && same(i, j)
            D.realization_date(i) = D.date(j);
            D.realization_rule(i) = "expiry_lookup_close";
            if j <= i, D.realization_rule(i) = "formation_close_fallback"; end
            break
        end
    end
end
% A level taken from the formation date itself (cleaning-script backward
% fallback) is a zero-return payoff in the source data; it is labeled, not
% altered. Any other realization must come after formation.
fb = D.realization_rule == "formation_close_fallback";
assert(all(D.realization_date(fb) == D.date(fb)) && all(D.realization_date(~fb) > D.date(~fb)), ...
    'load_oos_panel:Timing', 'A payoff is realized before its own formation date.');
D.holding_days = days(D.realization_date - D.date);

r = -log(df.bond) ./ df.mat;
c = log(df.wrds_forward ./ df.SP500) ./ df.mat;
D.rf_rate = accumarray(g, r, [], @median);
D.carry_rate = accumarray(g, c, [], @median);
D.rf_rate_spread = accumarray(g, r, [], @range);
D.rf_hold = exp(D.rf_rate .* D.holding_days) - 1;
D.carry_hold = exp(D.carry_rate .* D.holding_days) - 1;

% Thin-date filter (manuscript: before the 70/30 split).
keepDate = D.J >= oos.protocol.min_options;
panel = struct();
panel.n_dates_unfiltered = T0;
panel.thin_dates = D.date(~keepDate);
panel.dates_table = D(keepDate, :);
panel.df = df(ismember(df.date, panel.dates_table.date), :);
panel.dates = panel.dates_table.date;
panel.T = numel(panel.dates);
panel.half_spread = (panel.df.ask - panel.df.bid) ./ (2 * panel.df.SP500 .* panel.df.bond);
assert(all(panel.half_spread >= 0), 'load_oos_panel:Spread', 'Negative half-spread.');

end

function x = one_per_date(v, g, name)
%% Value that must be constant within each date.
lo = accumarray(g, v, [], @min);
hi = accumarray(g, v, [], @max);
assert(all(abs(hi - lo) <= 1e-12 * max(1, abs(hi))), 'load_oos_panel:NotConstant', ...
    '%s varies within a date.', name);
x = lo;
end
