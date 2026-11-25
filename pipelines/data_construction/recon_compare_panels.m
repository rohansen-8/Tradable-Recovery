function S = recon_compare_panels(reconRoot, outTables, outRows, candFile, dateMax)
%% Compare reconstructed panels with SDF/data/{hz}/df.csv: dates, rows, contract fields and numerical differences.
%
% S = recon_compare_panels(reconRoot, outTables, outRows, candFile, dateMax)
%   reconRoot  folder with {hz}/df.mat and merged_keys.mat (from recon_build_panels)
%   outTables  folder for shareable aggregate CSVs (no source rows)
%   outRows    folder for row-level discrepancy files (licensed-derived)
%   candFile   extraction .mat holding R (raw rows of the distributed panels' contracts)
%   dateMax    optional: compare only dates <= dateMax (test on a date slice)
%
% Rows are matched on (date, optionid). Unmatched rows are classified by the
% first applicable reason; matched rows are compared field by field, split
% at the 2023-01-01 forward splice. The distributed panels are read only.

c = recon_config();
if nargin < 5, dateMax = datetime(9999, 1, 1); end
for d = {outTables, outRows}, if ~isfolder(d{1}), mkdir(d{1}); end, end
X = load(candFile, 'R'); Raw = X.R;
M = load(fullfile(reconRoot, 'merged_keys.mat'), 'mergedKeys', 'dupKeys');
splice = datetime(2023, 1, 1);
fields = {'strike', 'abs'; 'flag', 'abs'; 'mat', 'abs'; 'am_settlement', 'abs'; 'bid', 'abs'; 'ask', 'abs'; ...
    'mid_price', 'abs'; 'volume', 'abs'; 'oi', 'abs'; 'iv', 'abs'; 'wrds_forward', 'log'; 'frwd_moneyness', 'log'; ...
    'bond', 'abs'; 'R_f', 'abs'; 'SP500', 'log'; 'sp500_shifted', 'log'; 'SP500_ret', 'abs'; 'VIX', 'abs'; ...
    'VIX_lagged', 'abs'; 'opt_gross_payoff', 'abs'; 'OPT_NET_R_PAYOFF', 'abs'; 'weight', 'log'};
counts = {}; fstats = {}; unm = {}; dstats = {};
for h = 1:4
    hz = c.horizons{h};
    f = fullfile(c.panels_existing, hz, 'df.csv');
    o = detectImportOptions(f, 'VariableNamingRule', 'modify'); o = setvartype(o, 'date', 'datetime'); o = setvaropts(o, 'date', 'InputFormat', 'yyyy-MM-dd');
    o = setvartype(o, {'frwd_moneyness_label'}, 'string');
    E = readtable(f, o);
    L = load(fullfile(reconRoot, hz, 'df.mat'), 'P'); R = L.P;
    E = E(E.date <= dateMax, :); R = R(R.date <= dateMax, :);
    kE = key(E.date, E.optionid); kR = key(R.date, R.optionid);
    assert(numel(unique(kE)) == numel(kE) && numel(unique(kR)) == numel(kR), '%s: duplicate (date, optionid).', hz);
    dE = unique(E.date); dR = unique(R.date);
    [inR, iR] = ismember(kE, kR);
    % ---- counts
    per = {'all', 'pre2023', '2023plus'};
    for p = 1:3
        sE = pmask(E.date, p, splice); sR = pmask(R.date, p, splice);
        counts(end+1, :) = {hz, per{p}, numel(unique(E.date(sE))), numel(unique(R.date(sR))), ...
            numel(intersect(E.date(sE), R.date(sR))), sum(sE), sum(sR), sum(inR & sE), sum(~inR & sE), ...
            sum(~ismember(kR, kE) & sR)}; %#ok<AGROW>
    end
    % ---- matched rows: field differences
    A = E(inR, :); B = R(iR(inR), :);
    for p = 1:3
        s = pmask(A.date, p, splice);
        for j = 1:size(fields, 1)
            fn = fields{j, 1};
            if ~ismember(fn, A.Properties.VariableNames) || ~ismember(fn, B.Properties.VariableNames), continue, end
            a = double(A.(fn)(s)); b = double(B.(fn)(s));
            if strcmp(fields{j, 2}, 'log'), dd = log(b ./ a); else, dd = b - a; end
            ad = abs(dd);
            fstats(end+1, :) = {hz, per{p}, fn, fields{j, 2}, numel(dd), sum(dd == 0), ...
                sum(ad <= 1e-12 * max(1, abs(a))), sum(isnan(dd)), med(ad), pct(ad, 99), max([ad; 0]), mean(dd, 'omitnan')}; %#ok<AGROW>
        end
    end
    % exdate implied by the distributed panel (date + mat) vs recon exdate
    exE = A.date + days(A.mat);
    fstats(end+1, :) = {hz, 'all', 'exdate', 'days', height(A), sum(exE == B.exdate), sum(exE == B.exdate), 0, ...
        med(abs(days(B.exdate - exE))), NaN, max([abs(days(B.exdate - exE)); 0]), NaN}; %#ok<AGROW>
    fstats(end+1, :) = {hz, 'all', 'frwd_moneyness_label', 'equal', height(A), sum(A.frwd_moneyness_label == B.frwd_moneyness_label), ...
        sum(A.frwd_moneyness_label == B.frwd_moneyness_label), 0, NaN, NaN, NaN, NaN}; %#ok<AGROW>
    % ---- per-date comparison (row sets, weights)
    gd = unique(E.date); gd2 = unique(R.date);
    nE = arrayfun(@(x) sum(E.date == x), gd); nR = arrayfun(@(x) sum(R.date == x), gd2);
    [tfd, id] = ismember(gd, gd2);
    sameSet = false(numel(gd), 1);
    for k = find(tfd).'
        sameSet(k) = isequal(sort(E.optionid(E.date == gd(k))), sort(R.optionid(R.date == gd2(id(k)))));
    end
    sw = A.weight; rw = B.weight; dateSame = ismember(A.date, gd(sameSet));
    wd = abs(log(rw(dateSame) ./ sw(dateSame)));
    dstats(end+1, :) = {hz, numel(gd), sum(tfd), sum(sameSet), sum(tfd & ~sameSet), ...
        sum(nE(tfd) ~= nR(id(tfd))), max([wd; 0]), med(wd)}; %#ok<AGROW>
    % ---- unmatched rows: classify
    oE = E(~inR, {'date', 'optionid', 'mat', 'strike', 'flag', 'frwd_moneyness_label'});
    oE.reason = classify_existing_only(oE, dR, Raw, M, c);
    oR = R(~ismember(kR, kE), {'date', 'optionid', 'mat', 'strike', 'flag', 'frwd_moneyness_label', 'frwd_log_moneyness'});
    oR.reason = classify_recon_only(oR, dE, E, c);
    oE.side = repmat("existing_only", height(oE), 1); oR.side = repmat("reconstructed_only", height(oR), 1);
    oR.frwd_log_moneyness = [];
    U = [oE; oR]; U.maturity = repmat(string(hz), height(U), 1);
    U.period = repmat("pre2023", height(U), 1); U.period(U.date >= splice) = "2023plus";
    writetable(U, fullfile(outRows, [hz '_unmatched_rows.csv']));
    if ~isempty(U)
        [gu, K] = findgroups(U(:, {'maturity', 'side', 'period', 'reason'}));
        K.n_rows = splitapply(@numel, U.optionid, gu);
        K.n_dates = splitapply(@(x) numel(unique(x)), U.date, gu);
        unm{end+1} = K; %#ok<AGROW>
    end
    % row-level field differences for matched rows (licensed-derived)
    Dm = table(A.date, A.optionid, A.mat, A.flag, log(B.wrds_forward ./ A.wrds_forward), B.bid - A.bid, B.ask - A.ask, ...
        log(B.SP500 ./ A.SP500), log(B.sp500_shifted ./ A.sp500_shifted), B.OPT_NET_R_PAYOFF - A.OPT_NET_R_PAYOFF, ...
        log(B.weight ./ A.weight), 'VariableNames', {'date', 'optionid', 'mat', 'flag', 'dlog_forward', 'd_bid', 'd_ask', ...
        'dlog_sp500', 'dlog_sp500_shifted', 'd_payoff', 'dlog_weight'});
    writetable(Dm, fullfile(outRows, [hz '_matched_differences.csv']));
end
S.counts = cell2table(counts, 'VariableNames', {'maturity', 'period', 'dates_existing', 'dates_recon', 'dates_both', ...
    'rows_existing', 'rows_recon', 'rows_matched', 'rows_existing_only', 'rows_recon_only'});
S.fields = cell2table(fstats, 'VariableNames', {'maturity', 'period', 'field', 'metric', 'n', 'n_identical', ...
    'n_within_1e12', 'n_nan', 'median_abs_diff', 'p99_abs_diff', 'max_abs_diff', 'mean_diff'});
S.unmatched = vertcat(unm{:});
if isempty(S.unmatched)
    S.unmatched = table(strings(0, 1), strings(0, 1), strings(0, 1), strings(0, 1), zeros(0, 1), zeros(0, 1), ...
        'VariableNames', {'maturity', 'side', 'period', 'reason', 'n_rows', 'n_dates'});
end
S.dates = cell2table(dstats, 'VariableNames', {'maturity', 'dates_existing', 'dates_in_both', 'dates_same_rowset', ...
    'dates_different_rowset', 'dates_different_count', 'max_abs_dlog_weight_same_rowset', 'median_abs_dlog_weight_same_rowset'});
writetable(S.counts, fullfile(outTables, 'stage1_compare_counts.csv'));
writetable(S.fields, fullfile(outTables, 'stage1_compare_fields.csv'));
writetable(S.unmatched, fullfile(outTables, 'stage1_compare_unmatched_reasons.csv'));
writetable(S.dates, fullfile(outTables, 'stage1_compare_dates.csv'));
end

% =====================================================================
function r = classify_existing_only(U, dR, Raw, M, c)
%% Why a distributed-panel row is absent from the reconstruction (first applicable reason).
r = repmat("other", height(U), 1);
k = key(U.date, U.optionid);
kRaw = key(Raw.date, Raw.optionid);
[inRaw, iRaw] = ismember(k, kRaw);
inDup = ismember(k, key(M.dupKeys.date, M.dupKeys.optionid));
[inMerged, iM] = ismember(k, key(M.mergedKeys.date, M.mergedKeys.optionid));
for i = 1:height(U)
    if ~ismember(U.date(i), dR), r(i) = "date_not_in_reconstruction"; end
    if ~inRaw(i), r(i) = r(i) + "|not_in_current_wrds_file"; continue, end
    x = Raw(iRaw(i), :);
    tests = {x.date >= c.clean_dates(1) && x.date <= c.clean_dates(2), 'date_range'; ...
        x.best_bid > double(x.best_offer > 0), 'bid_le_(ask>0)'; x.best_bid <= x.best_offer, 'bid_gt_ask'; ...
        x.open_interest > 0, 'oi_zero'; x.volume > 0, 'volume_zero'; ~isnan(x.impl_volatility), 'iv_missing'};
    bad = find(~[tests{:, 1}], 1);
    if ~isempty(bad), r(i) = r(i) + "|fails_filter:" + tests{bad, 2}; continue, end
    if inDup(i), r(i) = r(i) + "|removed_as_duplicate"; continue, end
    if inMerged(i) && M.mergedKeys.frwd_moneyness_label(iM(i)) == "ITF", r(i) = r(i) + "|itf_in_reconstruction"; continue, end
    if ~inMerged(i), r(i) = r(i) + "|dropped_at_merge(vix_or_forward)"; continue, end
    r(i) = r(i) + "|passes_cleaning_not_selected";
end
r = strip(r, 'left', '|');
r(r == "") = "other";
end

function r = classify_recon_only(U, dE, E, c)
%% Why a reconstructed row is absent from the distributed panel (first applicable reason).
r = repmat("other", height(U), 1);
r(~ismember(U.date, dE)) = "date_not_in_existing";
if isempty(E), return, end
% label under the distributed panel's forward for the same (date, mat)
[g, gd, gm] = findgroups(E.date, E.mat);
fE = splitapply(@(x) x(1), E.wrds_forward, g);
[tf, loc] = ismember([datenum(U.date) U.mat], [datenum(gd) gm], 'rows'); %#ok<DATNM>
lm = nan(height(U), 1);
lm(tf) = log(U.strike(tf) ./ fE(loc(tf)));
itf = (lm > c.tol_atm & U.flag == 0) | (lm < -c.tol_atm & U.flag == 1);
r(r == "other" & itf) = "itf_under_existing_forward";
r(r == "other" & ~tf) = "maturity_not_in_existing_on_date";
end

function m = pmask(d, p, splice)
%% Period mask: all, before the 2023 splice, from the splice.
switch p
    case 1, m = true(size(d));
    case 2, m = d < splice;
    case 3, m = d >= splice;
end
end

function k = key(d, id)
%% Exact scalar key for (date, optionid).
k = floor(datenum(d)) * 1e9 + id; %#ok<DATNM>
end

function q = med(x)
%% Median ignoring NaN; NaN if empty.
x = x(~isnan(x)); if isempty(x), q = NaN; else, q = median(x); end
end

function q = pct(x, p)
%% Nearest-rank percentile ignoring NaN.
x = sort(x(~isnan(x))); if isempty(x), q = NaN; else, q = x(max(1, ceil(p / 100 * numel(x)))); end
end
