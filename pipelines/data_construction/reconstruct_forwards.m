function [R, info] = reconstruct_forwards(pairs, F, Z, correction)
%% Forward price, discount factor and rate for requested (date, mat) pairs from OptionMetrics data.
%
% [R, info] = reconstruct_forwards(pairs, F, Z, correction)
%
% Implements the forward construction of the authors' Python data pipeline
% (OptPortOpt_V2 build_forwards.py):
%   * forward quotes (fwdprd): mat = expiration - date in calendar days,
%     mat > 0 only; AM- and PM-settled quotes with the same (date, mat) are
%     averaged (arithmetic mean);
%   * forward for a requested mat: linear interpolation in mat across the
%     date's quotes, held constant beyond the shortest/longest quoted mat;
%   * zero rate (zerocd, percent -> decimal): the curve of the date or the
%     latest earlier curve (at most 7 days old), linear in days, held
%     constant beyond the curve's ends;
%   * bond = exp(-r mat/365);  R_f = -log(bond)/mat * 252 (cleaning.py).
%
% pairs: table(date datetime, mat double). F: table(date, expiration,
% AMSettlement, ForwardPrice). Z: table(date, days, rate[percent]).
% correction: 'none' (source data as delivered), 'exclude_scale_errors'
% (diagnostic only: drops forward quotes below half or above twice the
% median quote of the same date before averaging), or 'narrow_20231019'
% (corrected panels: drops exactly the one confirmed corrupt quote, AM-settled,
% date 2023-10-19, expiration 2023-10-20, so the PM quote of the same date and
% expiration is used alone). The flagged quotes are reported in info.flagged
% under every setting.

if nargin < 4, correction = 'none'; end
assert(any(strcmp(correction, {'none', 'exclude_scale_errors', 'narrow_20231019'})), 'Unknown correction.');
maxGap = 7;

% Forward quotes, scale-error screen (reported always, applied on request)
F.mat = days(F.expiration - F.date);
F = F(F.mat > 0, :);
assert(all(F.ForwardPrice > 0), 'Non-positive forward price.');
[gd, ~] = findgroups(F.date);
med = splitapply(@median, F.ForwardPrice, gd);
ratio = F.ForwardPrice ./ med(gd);
bad = ratio < 0.5 | ratio > 2;
info.flagged = F(bad, {'date', 'expiration', 'AMSettlement', 'ForwardPrice', 'mat'});
info.flagged.ratio_to_date_median = ratio(bad);
if strcmp(correction, 'exclude_scale_errors'), F = F(~bad, :); end
if strcmp(correction, 'narrow_20231019')
    hit = F.date == datetime(2023, 10, 19) & F.expiration == datetime(2023, 10, 20) & F.AMSettlement == 1;
    assert(sum(hit) == 1 && all(bad(hit)), 'The 2023-10-19 corrupt quote is not where expected.');
    assert(sum(bad & F.date >= datetime(2000, 1, 1)) == 1, 'Unexpected additional scale errors in 2000+.');
    F = F(~hit, :);
end

% AM/PM collapse: one quote per (date, mat); findgroups returns sorted keys
[g, qd, qm] = findgroups(F.date, F.mat);
qf = splitapply(@mean, F.ForwardPrice, g);
qn = accumarray(g, 1);
info.n_quote_pairs = numel(qf);
info.n_ampm_averaged = sum(qn > 1);

% Zero curve in decimals, sorted by (date, days)
Z = Z(~isnan(Z.days) & ~isnan(Z.rate), :);
Z = sortrows(Z, {'date', 'days'});
assert(median(Z.rate) > 0.5, 'zerocd rates are expected in percent.');
Z.rate = Z.rate / 100;
[zd, zFirst] = unique(Z.date, 'first');
[~, zLast] = unique(Z.date, 'last');
[qdu, qFirst] = unique(qd, 'first');
[~, qLast] = unique(qd, 'last');

pairs = sortrows(pairs(:, {'date', 'mat'}), {'date', 'mat'});
n = height(pairs);
fwd = nan(n, 1); r = nan(n, 1); exact = false(n, 1); nAvg = zeros(n, 1);
clamped = false(n, 1); zcDate = NaT(n, 1); zcGap = nan(n, 1); noQuote = false(n, 1);
[ud, ~, ip] = unique(pairs.date);
iz = discretize(ud, [zd; datetime(9999, 1, 1)]);   % latest curve on or before each date
for k = 1:numel(ud)
    rows = find(ip == k);
    m = pairs.mat(rows);
    j = find(qdu == ud(k), 1);
    if isempty(j)
        noQuote(rows) = true;
    else
        qi = qFirst(j):qLast(j);
        x = qm(qi); y = qf(qi);
        assert(numel(x) >= 2, 'Fewer than 2 forward quotes on %s.', string(ud(k)));
        mc = min(max(m, x(1)), x(end));
        fwd(rows) = interp1(x, y, mc, 'linear');
        clamped(rows) = mc ~= m;
        [exact(rows), loc] = ismember(m, x);
        nAvg(rows(exact(rows))) = qn(qi(loc(exact(rows))));
    end
    assert(~isnan(iz(k)), 'No zero curve on or before %s.', string(ud(k)));
    gap = days(ud(k) - zd(iz(k)));
    assert(gap <= maxGap, 'Zero curve stale by %d days at %s.', gap, string(ud(k)));
    zi = zFirst(iz(k)):zLast(iz(k));
    xd = Z.days(zi); yr = Z.rate(zi);
    r(rows) = interp1(xd, yr, min(max(m, xd(1)), xd(end)), 'linear');
    zcDate(rows) = zd(iz(k)); zcGap(rows) = gap;
end
bond = exp(-r .* pairs.mat / 365);
R = pairs;
R.wrds_forward = fwd; R.bond = bond; R.R_f = -log(bond) ./ pairs.mat * 252; R.rate = r;
R.exact_quote = exact; R.n_quotes_averaged = nAvg; R.fwd_clamped = clamped;
R.no_forward_quote = noQuote; R.zc_date = zcDate; R.zc_gap_days = zcGap;
info.correction = correction;
end
