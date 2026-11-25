function out = recon_build_panels(mode, correction, outRoot)
%% Merge exogenous data, de-duplicate, apply the four maturity rules, add weights; write panels + manifests.
%
% out = recon_build_panels(mode, correction, outRoot)
%   mode       'full' (extracted/options_candidates.mat) or 'smoke' (row-limited test extract, extracted/smoke/...)
%   correction 'none' (forward quotes as delivered; the baseline) or
%              'exclude_scale_errors' (diagnostic; see reconstruct_forwards)
%   outRoot    output folder holding {quarterly,monthly,weekly,daily}/; default
%              baseline_replication for ('full','none'). Existing folders are never overwritten.
%
% Implements the panel construction of the authors' Python data pipeline
% (OptPortOpt_V2): cleaning.py (merge, VIX/forward filters, keep=False
% de-duplication on (date, strike, mat, flag), labels, mid price) and the
% preparing_* scripts (date selection, maturity choice, payoff-date lookup,
% lagged VIX, payoffs, ITF removal, daily screens). Weights: equal-step
% trapezoid rule (utils/helpers.calculate_trapezoidal_weights) on each date's
% final rows. Columns not constructed (not used by the estimator): cay,
% credit_spread, put_volume. S&P 500 levels: OptionMetrics secprd close/open.

c = recon_config();
if nargin < 2, correction = 'none'; end
if nargin < 3
    assert(strcmp(mode, 'full') && strcmp(correction, 'none'), 'outRoot required for smoke/diagnostic builds.');
    outRoot = c.baseline;
end
for h = 1:4, assert(~isfolder(fullfile(outRoot, c.horizons{h})), 'Output %s/%s exists; refusing to overwrite.', outRoot, c.horizons{h}); end
candDir = c.extracted; if strcmp(mode, 'smoke'), candDir = fullfile(c.extracted, 'smoke'); end
L = load(fullfile(candDir, 'options_candidates.mat'), 'C', 'info');
D = L.C; xinfo = L.info;
t0 = tic;

% ---- Exogenous: SPX (secprd) left-joined with VIX (Cboe), then onto options by date
csv = extract_small(c, {'spx', 'vix', 'forwards', 'zerocd'});
o = detectImportOptions(csv.spx); o = setvartype(o, 'date', 'datetime'); o = setvaropts(o, 'date', 'InputFormat', 'yyyy-MM-dd');
S = readtable(csv.spx, o);
o = detectImportOptions(csv.vix); o = setvartype(o, 'DATE', 'datetime'); o = setvaropts(o, 'DATE', 'InputFormat', 'MM/dd/yyyy');
V = readtable(csv.vix, o);
assert(numel(unique(S.date)) == height(S) && numel(unique(V.DATE)) == height(V), 'Duplicate exogenous dates.');
[tf, iv] = ismember(S.date, V.DATE);
S.VIX = nan(height(S), 1); S.VIX_OPEN = S.VIX;
S.VIX(tf) = V.CLOSE(iv(tf)); S.VIX_OPEN(tf) = V.OPEN(iv(tf));
[tf, is] = ismember(D.date, S.date);
D.SP500 = nan(height(D), 1); D.SP500_OPEN = D.SP500; D.VIX = D.SP500; D.VIX_OPEN = D.SP500;
D.SP500(tf) = S.close(is(tf)); D.SP500_OPEN(tf) = S.open(is(tf));
D.VIX(tf) = S.VIX(is(tf)); D.VIX_OPEN(tf) = S.VIX_OPEN(is(tf));
cnt.candidates = height(D);
cnt.no_spx_row = sum(~tf);

% ---- Forwards and discount factors (reconstruct_forwards) on every candidate (date, mat)
o = detectImportOptions(csv.forwards);
o.SelectedVariableNames = {'date', 'expiration', 'AMSettlement', 'ForwardPrice'};
o = setvartype(o, {'date', 'expiration'}, 'datetime'); o = setvaropts(o, {'date', 'expiration'}, 'InputFormat', 'yyyy-MM-dd');
F = readtable(csv.forwards, o);
o = detectImportOptions(csv.zerocd); o = setvartype(o, 'date', 'datetime'); o = setvaropts(o, 'date', 'InputFormat', 'yyyy-MM-dd');
Z = readtable(csv.zerocd, o);
pairs = unique(D(:, {'date', 'mat'}));
[Fw, finfo] = reconstruct_forwards(pairs, F, Z, correction);
[~, loc] = ismember([datenum(D.date) D.mat], [datenum(Fw.date) Fw.mat], 'rows'); %#ok<DATNM>
assert(all(loc > 0), 'Forward pair lookup failed.');
D.wrds_forward = Fw.wrds_forward(loc); D.bond = Fw.bond(loc); D.R_f = Fw.R_f(loc);
D.fwd_exact_quote = Fw.exact_quote(loc); D.fwd_clamped = Fw.fwd_clamped(loc);

% ---- cleaning.py filters needing merged data, then keep=False de-duplication
D = D(~isnan(D.VIX), :);          cnt.after_vix = height(D);
D = D(~isnan(D.wrds_forward), :); cnt.after_forward = height(D);
g = findgroups(D.date, D.strike, D.mat, D.flag);
n = accumarray(g, 1); dup = n(g) > 1;
dupKeys = D(dup, {'date', 'optionid', 'mat', 'strike', 'flag', 'am_settlement'});
D = D(~dup, :);                   cnt.after_dedup = height(D); cnt.dup_rows_removed = sum(dup);

% ---- Derived columns (cleaning.py)
D = renamevars(D, {'best_bid', 'best_offer', 'open_interest', 'impl_volatility'}, {'bid', 'ask', 'oi', 'iv'});
D.theta = D.theta / 252;
D.moneyness = D.strike ./ D.SP500;
D.log_moneyness = log(D.strike ./ D.SP500);
D.frwd_moneyness = D.strike ./ D.wrds_forward;
D.frwd_log_moneyness = log(D.strike ./ D.wrds_forward);
D.moneyness_label = label(D.log_moneyness, D.flag, c.tol_atm, ["ITM" "ATM" "OTM"]);
D.frwd_moneyness_label = label(D.frwd_log_moneyness, D.flag, c.tol_atm, ["ITF" "ATF" "OTF"]);
D.mid_price = (D.ask + D.bid) / 2;
D.bid_ask_spread = (D.ask - D.bid) ./ D.mid_price;
D.embedded_lev = abs(D.delta .* D.SP500 ./ D.mid_price);
D = sortrows(D, {'date', 'mat', 'strike', 'flag'});
mkdir(outRoot);
mergedKeys = D(:, {'date', 'optionid', 'mat', 'frwd_moneyness_label', 'wrds_forward'});
save(fullfile(outRoot, 'merged_keys.mat'), 'mergedKeys', 'dupKeys', 'cnt', '-v7.3');

% ---- Maturity panels
builders = struct('quarterly', @(D) build_mq(D, c.q, c), 'monthly', @(D) build_mq(D, c.m, c), ...
    'weekly', @(D) build_weekly(D, c.w, c), 'daily', @(D) build_daily(D, c.d, c));
codeFiles = {'recon_config', 'recon_extract_options', 'recon_build_panels', 'reconstruct_forwards', 'recon_sha'};
code = struct(); for f = codeFiles, code.(f{1}) = recon_sha(which(f{1})); end
inputs = struct('options_zip', xinfo.input_zip_sha256);
for f = {'forwards', 'zerocd', 'spx', 'vix'}, inputs.([f{1} '_zip']) = recon_sha(fullfile(c.raw, c.zip.(f{1}))); end
[~, gitHead] = system(sprintf('git -C "%s" rev-parse HEAD', fileparts(which('recon_build_panels'))));
[~, gitDirty] = system(sprintf('git -C "%s" status --porcelain -- .', fileparts(which('recon_build_panels'))));
out = struct('counts_merge', cnt, 'forward_info', rmfield(finfo, 'flagged'));
for h = 1:4
    hz = c.horizons{h};
    [P, steps] = builders.(hz)(D);
    P.weight = equal_step_weights(P.date, P.frwd_moneyness);
    P = sortrows(P, {'date', 'frwd_moneyness', 'optionid'});
    P.date.Format = 'yyyy-MM-dd'; P.exdate.Format = 'yyyy-MM-dd'; P.payoff_date.Format = 'yyyy-MM-dd';
    d = fullfile(outRoot, hz); mkdir(d);
    writetable(P, fullfile(d, 'df.csv'));
    save(fullfile(d, 'df.mat'), 'P', '-v7.3');
    man = struct('maturity', hz, 'created', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
        'mode', mode, 'forward_correction', correction, 'matlab', version, ...
        'git_head', strtrim(gitHead), 'git_uncommitted_changes', ~isempty(strtrim(gitDirty)), ...
        'code_sha256', code, 'inputs_sha256', inputs, 'extraction', xinfo, ...
        'rules', c.(hz(1)), 'cleaning', struct('dates', c.clean_dates, 'tol_atm', c.tol_atm, ...
        'bid_filter', 'bid > (ask > 0)  [as parsed from cleaning.py]', 'dedup', 'drop all rows of duplicated (date,strike,mat,flag)'), ...
        'weights', 'equal-step trapezoid: h=(max m - min m)/(n-1), h/2 at the two extreme points, h elsewhere', ...
        'sp500_source', 'OptionMetrics secprd (secid 108105) close/open', 'vix_source', 'Cboe VIX_History CLOSE/OPEN', ...
        'counts_merge', cnt, 'counts_steps', steps, 'n_rows', height(P), 'n_dates', numel(unique(P.date)), ...
        'first_date', char(min(P.date)), 'last_date', char(max(P.date)), 'df_csv_sha256', recon_sha(fullfile(d, 'df.csv')));
    fid = fopen(fullfile(d, 'manifest.json'), 'w'); fprintf(fid, '%s', jsonencode(man, 'PrettyPrint', true)); fclose(fid);
    out.(hz) = man;
    fprintf('%-9s %7d rows %5d dates  %s .. %s\n', hz, height(P), man.n_dates, man.first_date, man.last_date);
end
out.seconds = toc(t0);
end

% =====================================================================
function [P, s] = build_mq(D, r, c)
%% Quarterly/monthly: Hu-Liu collection dates, mat window, payoff at the next sample collection date.
tf = third_fridays(r.dates, r.months);
expi = tf; sat = tf < c.saturday_expiry_before; expi(sat) = tf(sat) + 1;
coll = expi + 1;
while true
    we = weekday(coll) == 1 | weekday(coll) == 7;
    if ~any(we), break; end
    coll(we) = coll(we) + 1;
end
coll = coll(coll <= r.dates(2));                 % pd.bdate_range(START, END)
P = D(ismember(D.date, coll) & D.mat >= r.mat(1) & D.mat <= r.mat(2), :);
s.collection_dates = numel(coll); s.dates_with_options = numel(unique(P.date)); s.rows_selected = height(P);
[ud, first, ig] = unique(P.date);
S0 = P.SP500(first);
nxt = [S0(2:end); NaN]; nd = [ud(2:end); NaT];
P.sp500_shifted = nxt(ig); P.payoff_date = nd(ig);
P = P(~isnan(P.sp500_shifted), :);               s.rows_after_payoff = height(P);
[~, first, ig] = unique(P.date);
V0 = P.VIX(first); lag = [NaN; V0(1:end-1)];
P.VIX_lagged = lag(ig);
P.VIX_ret = log(P.VIX) - log(P.VIX_lagged);
P.VIX_gross_payoff = P.VIX ./ P.VIX_lagged;
P = P(~isnan(P.VIX_lagged), :);                  s.rows_after_vix_lag = height(P);
P.SP500_log_ret = log(P.sp500_shifted) - log(P.SP500);
P.SP500_ret = P.sp500_shifted ./ P.SP500 - 1;
P.SP500_payoff = P.sp500_shifted - P.SP500;
P = payoffs(P);
P = P(P.frwd_moneyness_label ~= "ITF", :);       s.rows_after_itf = height(P);
end

function [P, s] = build_weekly(D, r, c) %#ok<INUSD>
%% Weekly: Fridays, mat 5-9 closest to 7 (shorter on ties), payoff at date+mat (7 for 8-9) among sample dates.
P = D(weekday(D.date) == 6 & D.date >= r.dates(1) & D.date <= r.dates(2) & D.mat >= r.mat(1) & D.mat <= r.mat(2), :);
P = closest_mat(P, r.target);                    s.rows_selected = height(P); s.dates_selected = numel(unique(P.date));
P.mat_weekly = P.mat;
P.expiry_date = P.date + days(P.mat);
P.effective_maturity = P.mat; P.effective_maturity(ismember(P.mat, [8 9])) = 7;
P.sp500_lookup_date = P.date + days(P.effective_maturity);
[ud, first] = unique(P.date);
[P.sp500_shifted, P.payoff_date] = lookup(P.sp500_lookup_date, ud, P.SP500(first));
P = P(~isnan(P.sp500_shifted), :);               s.rows_after_payoff = height(P);
[ud, first, ig] = unique(P.date);
V0 = P.VIX(first); lag = [NaN; V0(1:end-1)];
P.VIX_lagged = lag(ig);
P.VIX_shifted = lookup(P.sp500_lookup_date, ud, V0);
P.VIX_gross_payoff = P.VIX ./ P.VIX_lagged;
P.VIX_gross_payoff_lookahead = P.VIX_shifted ./ P.VIX;
P = P(~isnan(P.VIX_lagged) & ~isnan(P.VIX_shifted), :); s.rows_after_vix = height(P);
P = returns(P);
P = payoffs(P);
P = P(P.frwd_moneyness_label ~= "ITF", :);       s.rows_after_itf = height(P);
end

function [P, s] = build_daily(D, r, c) %#ok<INUSD>
%% Daily: weekdays, mat 1-4 closest to 1, >=10 options, payoff at expiry (Fri for weekend), screens after ITF.
wd = weekday(D.date);
P = D(wd >= 2 & wd <= 6 & D.date >= r.dates(1) & D.date <= r.dates(2) & D.mat >= r.mat(1) & D.mat <= r.mat(2), :);
P = closest_mat(P, r.target);                    s.rows_selected = height(P);
P.mat_daily = P.mat;
P = min_count(P, r.min_options);                 s.rows_after_count1 = height(P); s.dates_after_count1 = numel(unique(P.date));
P.expiry_date = P.date + days(P.mat);
P.sp500_lookup_date = P.expiry_date;
ew = weekday(P.expiry_date);
P.sp500_lookup_date(ew == 7) = P.expiry_date(ew == 7) - 1;
P.sp500_lookup_date(ew == 1) = P.expiry_date(ew == 1) - 2;
P.effective_maturity = days(P.sp500_lookup_date - P.date);
[ud, first] = unique(P.date);
[P.sp500_shifted, P.payoff_date] = lookup(P.sp500_lookup_date, ud, P.SP500(first));
P = P(~isnan(P.sp500_shifted), :);               s.rows_after_payoff = height(P);
[ud, first] = unique(P.date);
P.VIX_shifted = lookup(P.sp500_lookup_date, ud, P.VIX(first));
P.VIX_gross_payoff = P.VIX_shifted - P.VIX;
P.VIX_gross_return = P.VIX_shifted ./ P.VIX;
P = P(~isnan(P.VIX_shifted), :);                 s.rows_after_vix = height(P);
P = returns(P);
P = payoffs(P);
P = P(P.frwd_moneyness_label ~= "ITF", :);       s.rows_after_itf = height(P);
P = min_count(P, r.min_options);                 s.rows_after_count2 = height(P);
[ud, ~, ig] = unique(P.date);
sd = accumarray(ig, P.frwd_log_moneyness, [numel(ud) 1], @std);   % pandas std, ddof = 1
P = P(sd(ig) > 0, :);                            s.rows_after_std = height(P); s.dates_final = numel(unique(P.date));
end

% =====================================================================
function P = closest_mat(P, target)
%% Keep, per date, the maturity closest to target; the shorter one on ties.
[ud, ~, ig] = unique(P.date);
dist = abs(P.mat - target);
mind = accumarray(ig, dist, [numel(ud) 1], @min);
c = dist == mind(ig);
best = accumarray(ig(c), P.mat(c), [numel(ud) 1], @min);
P = P(P.mat == best(ig), :);
end

function P = min_count(P, k)
%% Keep dates with at least k rows.
[~, ~, ig] = unique(P.date);
n = accumarray(ig, 1);
P = P(n(ig) >= k, :);
end

function [v, found] = lookup(q, dd, val)
%% Value on date q among sample dates dd: exact, then q+1..q+3, then q-1..q-2 (preparing_weekly/daily).
v = nan(size(q)); found = NaT(size(q)); todo = true(size(q));
for off = [0 1 2 3 -1 -2]
    [tf, loc] = ismember(q + off, dd);
    tf = tf & todo;
    v(tf) = val(loc(tf)); found(tf) = dd(loc(tf)); todo(tf) = false;
end
end

function P = returns(P)
%% Underlying return to the payoff date (weekly/daily scripts).
P.SP500_ret = P.sp500_shifted ./ P.SP500 - 1;
P.SP500_log_ret = log(P.sp500_shifted ./ P.SP500);
P.SP500_payoff = P.sp500_shifted - P.SP500;
end

function P = payoffs(P)
%% Option payoffs at S_T and the net payoff per unit of S_t: max(+-(S_T-K),0)/S_t - mid/(S_t bond).
ST = P.sp500_shifted; K = P.strike;
g = max(0, ST - K);
put = P.flag == 0;
g(put) = max(0, K(put) - ST(put));
P.opt_gross_payoff = g;
P.opt_gross_R_payoff = g ./ P.SP500;
P.opt_ret = g ./ P.mid_price .* (1 ./ P.bond) - 1;
P.opt_net_payoff = g - P.mid_price ./ P.bond;
P.OPT_NET_R_PAYOFF = P.opt_gross_R_payoff - P.mid_price ./ (P.SP500 .* P.bond);
end

function L = label(lm, flag, tol, names)
%% In/at/out-of-the-money label (cleaning.py): ITM if (lm>tol & put) | (lm<-tol & call); ATM if |lm|<=tol.
L = repmat(names(3), size(lm));
L(abs(lm) <= tol) = names(2);
L((lm > tol & flag == 0) | (lm < -tol & flag == 1)) = names(1);
L(isnan(lm)) = missing;
end

function w = equal_step_weights(d, m)
%% Equal-step trapezoid per date: h=(max-min)/(n-1); h/2 at the two extreme points.
w = nan(size(m));
[~, ~, ig] = unique(d);
for k = 1:max(ig)
    i = find(ig == k); n = numel(i);
    if n < 2, w(i) = 1; continue, end
    h = (max(m(i)) - min(m(i))) / (n - 1);
    assert(h > 0, 'Degenerate moneyness range.');
    wk = h * ones(n, 1);
    [~, o] = sort(m(i));
    wk(o([1 end])) = h / 2;
    w(i) = wk;
end
end

function tf = third_fridays(range, months)
%% Third Fridays (pd.date_range freq='WOM-3FRI') within range, restricted to the given months.
first = dateshift(range(1), 'start', 'month'):calmonths(1):range(2);
first = first(:);
fri = first + mod(6 - weekday(first), 7);       % first Friday (weekday 6)
tf = fri + 14;
tf = tf(tf >= range(1) & tf <= range(2) & ismember(month(tf), months));
end

function csv = extract_small(c, names)
%% Extract the single CSV of each small ZIP into extracted/ once (ZIPs untouched).
csv = struct();
for f = names
    z = fullfile(c.raw, c.zip.(f{1}));
    zf = java.util.zip.ZipFile(z); e = zf.entries(); nm = {};
    while e.hasMoreElements()
        n = char(e.nextElement().getName());
        if endsWith(n, '.csv') && ~startsWith(n, '__MACOSX'), nm{end+1} = n; end %#ok<AGROW>
    end
    zf.close();
    assert(numel(nm) == 1, '%s: expected one CSV.', z);
    target = fullfile(c.extracted, nm{1});
    if ~isfile(target)
        [st, o] = system(sprintf('unzip -o -j "%s" "%s" -d "%s"', z, nm{1}, c.extracted));
        assert(st == 0, 'unzip failed: %s', o);
    end
    csv.(f{1}) = target;
end
end
