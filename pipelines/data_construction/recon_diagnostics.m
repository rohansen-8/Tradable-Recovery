function S = recon_diagnostics(baseRoot, fwdDiagRoot, cmpRows, outDiag, outTables)
%% Option-panel reconstruction diagnostics: spacing-aware quadrature weights, expiration/settlement timing, forward anomaly and splice.
%
% S = recon_diagnostics(baseRoot, fwdDiagRoot, cmpRows, outDiag, outTables)
%   baseRoot     baseline panels ({hz}/df.mat)
%   fwdDiagRoot  panels rebuilt with forward correction 'exclude_scale_errors'
%   cmpRows      row-level comparison files from recon_compare_panels
%   outDiag      licensed-derived row/date-level outputs
%   outTables    shareable aggregate tables
% Lags are payoff date minus settlement trading date, in SPX trading days.
% Nothing here changes the baseline panels or their weights.

c = recon_config();
for d = {outDiag, outTables, fullfile(outDiag, 'quadrature_weights'), fullfile(outDiag, 'expiration')}
    if ~isfolder(d{1}), mkdir(d{1}); end
end
csv = fullfile(c.extracted, 'ewelqglgrmodbmtz.csv');
o = detectImportOptions(csv); o = setvartype(o, 'date', 'datetime'); o = setvaropts(o, 'date', 'InputFormat', 'yyyy-MM-dd');
SPX = readtable(csv, o);
qa = {}; ex = {}; exd = {}; fa = {}; fy = {};
for h = 1:4
    hz = c.horizons{h};
    L = load(fullfile(baseRoot, hz, 'df.mat'), 'P'); P = L.P;
    if isempty(P), continue, end                    % (row-limited test slices only)

    % ---- A. quadrature: equal-step (baseline) vs composite trapezoid on the observed spacing
    [ud, ~, ig] = unique(P.date);
    wt = nan(height(P), 1);
    nd = numel(ud); st = nan(nd, 9);
    for k = 1:nd
        i = find(ig == k); m = P.frwd_moneyness(i); n = numel(i);
        [ms, o2] = sort(m);
        if n >= 2
            dm = diff(ms);
            ws = ([dm; 0] + [0; dm]) / 2;          % composite trapezoid nodes weights
            wt(i(o2)) = ws;
        end
        we = P.weight(i); w2 = wt(i);
        x = P.OPT_NET_R_PAYOFF(i); q = P.mid_price(i) ./ (P.SP500(i) .* P.bond(i));
        lo = ms(1); hi = ms(end); exact = (hi^3 - lo^3) / 3;   % integral of m^2 over the date's range
        pos = diff(ms); pos = pos(pos > 0);
        if isempty(pos), st(k, 1) = n; continue, end
        st(k, :) = [n, max(pos) / min(pos), sum(diff(ms) == 0), sum(abs(w2 - we)) / sum(we), ...
            sum(we .* x) - sum(w2 .* x), (sum(w2 .* q) - sum(we .* q)) / sum(we .* q), ...
            sum(we .* m.^2) / exact - 1, sum(w2 .* m.^2) / exact - 1, max(abs(w2 - we)) / ((hi - lo) / max(n - 1, 1))];
    end
    W = table(P.date, P.optionid, P.frwd_moneyness, P.weight, wt, 'VariableNames', ...
        {'date', 'optionid', 'frwd_moneyness', 'weight_equal_step', 'weight_trapezoid_observed'});
    writetable(W, fullfile(outDiag, 'quadrature_weights', [hz '_weights.csv']));
    B = array2table(st, 'VariableNames', {'n', 'spacing_ratio_max_min', 'n_zero_spacing', 'rel_L1_weight_diff', ...
        'd_sum_w_payoff', 'rel_d_sum_w_price', 'rel_err_m2_equal_step', 'rel_err_m2_trapezoid', 'max_abs_dw_over_h'});
    B = [table(ud, 'VariableNames', {'date'}), B]; %#ok<AGROW>
    writetable(B, fullfile(outDiag, 'quadrature_weights', [hz '_by_date.csv']));
    qa(end+1, :) = {hz, nd, med(B.n), med(B.spacing_ratio_max_min), pct(B.spacing_ratio_max_min, 90), ...
        sum(B.n_zero_spacing > 0), med(B.rel_L1_weight_diff), pct(B.rel_L1_weight_diff, 90), max(B.rel_L1_weight_diff), ...
        med(abs(B.rel_d_sum_w_price)), pct(abs(B.rel_d_sum_w_price), 90), med(abs(B.d_sum_w_payoff)), ...
        std(arrayfun(@(k) sum(P.weight(ig == k) .* P.OPT_NET_R_PAYOFF(ig == k)), 1:nd)), ...
        med(abs(B.rel_err_m2_equal_step)), max(abs(B.rel_err_m2_equal_step)), med(abs(B.rel_err_m2_trapezoid)), ...
        max(abs(B.rel_err_m2_trapezoid))}; %#ok<AGROW>

    % ---- B. expiration and settlement timing
    % settlement trading date: last SPX trading date on or before exdate (Saturday AM expiries -> Friday)
    idx = discretize(P.exdate, [SPX.date; datetime(9999, 1, 1)]);
    sd = SPX.date(idx);
    pidx = discretize(P.payoff_date, [SPX.date; datetime(9999, 1, 1)]);
    lagCal = days(P.payoff_date - sd);
    lagTd = pidx - idx;                              % trading days between settlement and payoff dates
    E = table(P.date, P.optionid, P.exdate, weekday(P.exdate), P.am_settlement, sd, P.payoff_date, lagCal, lagTd, ...
        log(P.sp500_shifted ./ SPX.close(idx)), log(SPX.open(idx) ./ SPX.close(idx)), ...
        'VariableNames', {'date', 'optionid', 'exdate', 'exdate_weekday', 'am_settlement', 'settlement_trading_date', ...
        'payoff_date', 'payoff_minus_settlement_days', 'payoff_minus_settlement_tdays', 'dlog_payoff_close_vs_settle_close', ...
        'dlog_settle_open_vs_close'});
    writetable(E, fullfile(outDiag, 'expiration', [hz '_expiration_check.csv']));
    for am = [0 1]
        s = E.am_settlement == am;
        lg = E.payoff_minus_settlement_tdays;
        ex(end+1, :) = {hz, am, sum(s), numel(unique(E.date(s))), sum(s & lg < 0), sum(s & lg == 0), sum(s & lg == 1), ...
            sum(s & lg >= 2 & lg <= 5), sum(s & lg > 5), numel(unique(E.date(s & lg ~= 0))), ...
            sum(s & E.exdate_weekday == 7), sum(s & E.exdate_weekday == 6), sum(s & ~ismember(E.exdate_weekday, [6 7])), ...
            med(abs(E.dlog_payoff_close_vs_settle_close(s & E.payoff_minus_settlement_tdays ~= 0))), ...
            max([abs(E.dlog_payoff_close_vs_settle_close(s)); 0]), med(abs(E.dlog_settle_open_vs_close(s)))}; %#ok<AGROW>
    end
    % date-level: number of distinct expirations, payoff timing
    [ud2, ~, ig2] = unique(E.date);
    nex = accumarray(ig2, datenum(E.exdate), [], @(x) numel(unique(x))); %#ok<DATNM>
    minLag = accumarray(ig2, E.payoff_minus_settlement_tdays, [], @min);
    maxLag = accumarray(ig2, E.payoff_minus_settlement_tdays, [], @max);
    amShare = accumarray(ig2, E.am_settlement, [], @mean);
    Dd = table(ud2, nex, amShare, minLag, maxLag, 'VariableNames', {'date', 'n_expirations', 'am_share', ...
        'min_payoff_minus_settlement_tdays', 'max_payoff_minus_settlement_tdays'});
    [~, firstRow] = unique(E.date);
    Dd.payoff_date = E.payoff_date(firstRow);
    writetable(Dd, fullfile(outDiag, 'expiration', [hz '_by_date.csv']));
    exd(end+1, :) = {hz, numel(ud2), sum(nex > 1), sum(minLag < 0), sum(maxLag > 0), sum(minLag ~= 0 | maxLag ~= 0)}; %#ok<AGROW>

    % ---- C. forward anomaly: baseline vs 'exclude_scale_errors' rebuild
    L2 = load(fullfile(fwdDiagRoot, hz, 'df.mat'), 'P'); Q = L2.P;
    kP = key(P.date, P.optionid); kQ = key(Q.date, Q.optionid);
    [in, iq] = ismember(kP, kQ);
    dF = abs(log(Q.wrds_forward(iq(in)) ./ P.wrds_forward(in)));
    dW = abs(log(Q.weight(iq(in)) ./ P.weight(in)));
    ch = ~in; ch(in) = dF > 0 | dW > 1e-12;
    chD = unique([P.date(ch); Q.date(~ismember(kQ, kP))]);
    fa(end+1, :) = {hz, height(P), height(Q), sum(~in), sum(~ismember(kQ, kP)), sum(dF > 0), max([dF; 0]), ...
        sum(dW > 1e-12), numel(chD), strjoin(string(chD, 'yyyy-MM-dd'), ' ')}; %#ok<AGROW>

    % ---- forward splice: reconstructed vs distributed-panel forward by year (matched rows)
    f = fullfile(cmpRows, [hz '_matched_differences.csv']);
    o = detectImportOptions(f); o = setvartype(o, 'date', 'datetime'); o = setvaropts(o, 'date', 'InputFormat', 'yyyy-MM-dd');
    M = readtable(f, o);
    [gy, yr] = findgroups(year(M.date));
    fy{end+1} = table(repmat(string(hz), numel(yr), 1), yr, splitapply(@numel, M.dlog_forward, gy), ...
        splitapply(@(x) mean(x) * 1e4, M.dlog_forward, gy), splitapply(@(x) median(abs(x)) * 1e4, M.dlog_forward, gy), ...
        splitapply(@(x) max(abs(x)) * 1e4, M.dlog_forward, gy), splitapply(@(x) mean(x) * 1e4, M.d_payoff, gy) , ...
        'VariableNames', {'maturity', 'year', 'n_matched', 'mean_dlog_forward_bp', 'median_abs_dlog_forward_bp', ...
        'max_abs_dlog_forward_bp', 'mean_d_payoff_bp'}); %#ok<AGROW>
end
S.quadrature = cell2table(qa, 'VariableNames', {'maturity', 'dates', 'median_options_per_date', 'median_spacing_ratio', ...
    'p90_spacing_ratio', 'dates_with_duplicate_moneyness', 'median_rel_L1_weight_diff', 'p90_rel_L1_weight_diff', ...
    'max_rel_L1_weight_diff', 'median_abs_rel_d_price_integral', 'p90_abs_rel_d_price_integral', ...
    'median_abs_d_payoff_integral', 'sd_over_dates_payoff_integral', 'median_abs_relerr_m2_equal_step', ...
    'max_abs_relerr_m2_equal_step', 'median_abs_relerr_m2_trapezoid', 'max_abs_relerr_m2_trapezoid'});
S.expiration = cell2table(ex, 'VariableNames', {'maturity', 'am_settlement', 'rows', 'dates', 'rows_lag_negative', ...
    'rows_lag_0', 'rows_lag_1', 'rows_lag_2to5', 'rows_lag_gt5', 'dates_with_nonzero_lag', 'rows_exdate_saturday', ...
    'rows_exdate_friday', 'rows_exdate_other_weekday', 'median_abs_dlog_payoff_vs_settle_close_offset_rows', ...
    'max_abs_dlog_payoff_vs_settle_close', 'median_abs_dlog_open_vs_close_settlement_day'});
S.expiration_dates = cell2table(exd, 'VariableNames', {'maturity', 'dates', 'dates_multiple_expirations', ...
    'dates_some_payoff_before_settlement', 'dates_some_payoff_after_settlement', 'dates_any_offset'});
S.forward_anomaly = cell2table(fa, 'VariableNames', {'maturity', 'rows_baseline', 'rows_corrected', 'rows_only_baseline', ...
    'rows_only_corrected', 'rows_forward_changed', 'max_abs_dlog_forward', 'rows_weight_changed', 'dates_changed', 'changed_dates'});
S.forward_by_year = vertcat(fy{:});
writetable(S.quadrature, fullfile(outTables, 'stage1_quadrature_summary.csv'));
writetable(S.expiration, fullfile(outTables, 'stage1_expiration_summary.csv'));
writetable(S.expiration_dates, fullfile(outTables, 'stage1_expiration_dates_summary.csv'));
writetable(S.forward_anomaly, fullfile(outTables, 'stage1_forward_anomaly_summary.csv'));
writetable(S.forward_by_year, fullfile(outTables, 'stage1_forward_splice_by_year.csv'));
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
