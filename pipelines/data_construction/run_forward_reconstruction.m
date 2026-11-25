function S = run_forward_reconstruction()
%% Rebuild forwards, bonds and rates from WRDS data and compare them with the four panels.
%
% S = run_forward_reconstruction()
%
% Reads only the forward-price and zero-curve ZIPs in data_reconstructed/raw
% (their CSVs are extracted once to data_reconstructed/extracted; the ZIPs
% and the panels are never modified). Builds (i) the full 1996-2025 quote
% grid, every (date, mat > 0) of fwdprd, and (ii) every (date, mat) pair of
% the quarterly/monthly/weekly/daily panels, under correction 'none' and the
% diagnostic 'exclude_scale_errors'. Writes to data_reconstructed/forwards/
% and refuses to overwrite an existing output folder.

root = get_project_root();
rec = fullfile(root, 'data_reconstructed');
out = fullfile(rec, 'forwards');
assert(~isfolder(out), 'Output folder %s exists; refusing to overwrite.', out);
ext = fullfile(rec, 'extracted');
if ~isfolder(ext), mkdir(ext); end
src = struct('forwards', 'wrds_spx_forward_prices.zip', 'zerocd', 'wrds_zero_coupon_yield.zip');
csv = struct();
for f = fieldnames(src).'
    z = fullfile(rec, 'raw', src.(f{1}));
    names = list_zip(z);
    target = fullfile(ext, names{1});
    if ~isfile(target), unzip(z, ext); end
    csv.(f{1}) = target;
end

o = detectImportOptions(csv.forwards);
o.SelectedVariableNames = {'date', 'expiration', 'AMSettlement', 'ForwardPrice'};
o = setvartype(o, {'date', 'expiration'}, 'datetime');
o = setvaropts(o, {'date', 'expiration'}, 'InputFormat', 'yyyy-MM-dd');
F = readtable(csv.forwards, o);
o = detectImportOptions(csv.zerocd);
o = setvartype(o, 'date', 'datetime');
o = setvaropts(o, 'date', 'InputFormat', 'yyyy-MM-dd');
Z = readtable(csv.zerocd, o);

% Panel pairs; forward/bond/R_f must be constant within (date, mat)
hz = {'quarterly', 'monthly', 'weekly', 'daily'};
P = cell(1, 4);
for h = 1:4
    o = detectImportOptions(fullfile(root, 'data', hz{h}, 'df.csv'));
    o.SelectedVariableNames = {'date', 'mat', 'wrds_forward', 'bond', 'R_f'};
    o = setvartype(o, 'date', 'datetime');
    o = setvaropts(o, 'date', 'InputFormat', 'yyyy-MM-dd');
    T = readtable(fullfile(root, 'data', hz{h}, 'df.csv'), o);
    [g, d, m] = findgroups(T.date, T.mat);
    spread = splitapply(@(x) max(x) - min(x), T.wrds_forward, g);
    assert(max(spread) == 0, '%s: forward varies within a (date, mat) pair.', hz{h});
    P{h} = table(d, m, splitapply(@(x) x(1), T.wrds_forward, g), splitapply(@(x) x(1), T.bond, g), ...
        splitapply(@(x) x(1), T.R_f, g), repmat(string(hz{h}), numel(d), 1), ...
        'VariableNames', {'date', 'mat', 'fwd_panel', 'bond_panel', 'Rf_panel', 'maturity'});
end
P = vertcat(P{:});
pairs = unique(P(:, {'date', 'mat'}));
grid = unique(table(F.date, days(F.expiration - F.date), 'VariableNames', {'date', 'mat'}));
grid = grid(grid.mat > 0, :);

mkdir(out);
S = struct();
summ = {}; byYear = {}; worst = {};
for c = {'none', 'exclude_scale_errors'}
    [G, ~] = reconstruct_forwards(grid, F, Z, c{1});
    writetable(G, fullfile(out, ['quote_grid_' c{1} '.csv']));
    [R, info] = reconstruct_forwards(pairs, F, Z, c{1});
    writetable(R, fullfile(out, ['panel_pairs_' c{1} '.csv']));
    J = innerjoin(P, R, 'Keys', {'date', 'mat'});
    J.correction = repmat(string(c{1}), height(J), 1);
    J.period = repmat("1996-2022", height(J), 1);
    J.period(J.date >= datetime(2023, 1, 1)) = "2023-2025";
    J.dlog_fwd = log(J.wrds_forward ./ J.fwd_panel);
    J.dbond = J.bond - J.bond_panel;
    J.dRf = J.R_f - J.Rf_panel;
    J.year = year(J.date);
    summ{end+1} = stats(J, {'correction', 'maturity', 'period'}); %#ok<AGROW>
    byYear{end+1} = stats(J, {'correction', 'maturity', 'year'}); %#ok<AGROW>
    [~, ix] = sort(abs(J.dlog_fwd), 'descend', 'MissingPlacement', 'last');
    worst{end+1} = J(ix(1:25), {'correction', 'maturity', 'date', 'mat', 'fwd_panel', 'wrds_forward', ...
        'dlog_fwd', 'exact_quote', 'n_quotes_averaged', 'fwd_clamped', 'bond_panel', 'bond', 'dbond'}); %#ok<AGROW>
    S.(c{1}) = struct('info', info, 'J', J, 'grid', G);
end
S.summary = vertcat(summ{:});
S.by_year = vertcat(byYear{:});
S.worst = vertcat(worst{:});
writetable(S.summary, fullfile(out, 'comparison_summary.csv'));
writetable(S.by_year, fullfile(out, 'comparison_by_year.csv'));
writetable(S.worst, fullfile(out, 'comparison_worst.csv'));
writetable(S.none.info.flagged, fullfile(out, 'flagged_scale_error_quotes.csv'));
meta = struct('created', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
    'matlab', version, 'code_sha256', struct( ...
    'reconstruct_forwards', sha(which('reconstruct_forwards')), ...
    'run_forward_reconstruction', sha([mfilename('fullpath') '.m'])), ...
    'inputs_sha256', struct('forwards_zip', sha(fullfile(rec, 'raw', src.forwards)), ...
    'zerocd_zip', sha(fullfile(rec, 'raw', src.zerocd)), 'forwards_csv', sha(csv.forwards), ...
    'zerocd_csv', sha(csv.zerocd)), 'panel_sha256', struct('quarterly', sha(fullfile(root, 'data/quarterly/df.csv')), ...
    'monthly', sha(fullfile(root, 'data/monthly/df.csv')), 'weekly', sha(fullfile(root, 'data/weekly/df.csv')), ...
    'daily', sha(fullfile(root, 'data/daily/df.csv'))), 'n_grid_pairs', height(grid), 'n_panel_pairs', height(pairs));
fid = fopen(fullfile(out, 'run_meta.json'), 'w'); fprintf(fid, '%s', jsonencode(meta, 'PrettyPrint', true)); fclose(fid);
S.meta = meta;
end

function T = stats(J, keys)
%% Discrepancy summary of reconstructed minus panel values per group.
[g, K] = findgroups(J(:, keys));
a = @(x) abs(x);
T = K;
T.n = splitapply(@numel, J.dlog_fwd, g);
T.n_no_quote = splitapply(@(x) sum(x), J.no_forward_quote, g);
T.n_clamped = splitapply(@(x) sum(x), J.fwd_clamped, g);
T.med_abs_dlog_fwd = splitapply(@(x) median(a(x), 'omitnan'), J.dlog_fwd, g);
T.p99_abs_dlog_fwd = splitapply(@(x) prctile_(a(x), 99), J.dlog_fwd, g);
T.max_abs_dlog_fwd = splitapply(@(x) max(a(x)), J.dlog_fwd, g);
T.n_dlog_gt_1e3 = splitapply(@(x) sum(a(x) > 1e-3), J.dlog_fwd, g);
T.mean_dlog_fwd = splitapply(@(x) mean(x, 'omitnan'), J.dlog_fwd, g);
T.max_abs_dbond = splitapply(@(x) max(a(x)), J.dbond, g);
T.med_abs_dbond = splitapply(@(x) median(a(x), 'omitnan'), J.dbond, g);
T.max_abs_dRf = splitapply(@(x) max(a(x)), J.dRf, g);
end

function q = prctile_(x, p)
%% Percentile by sorting (base MATLAB; nearest-rank).
x = sort(x(~isnan(x)));
if isempty(x), q = NaN; return, end
q = x(max(1, ceil(p / 100 * numel(x))));
end

function names = list_zip(z)
%% CSV entries of a ZIP (macOS resource forks excluded).
zf = java.util.zip.ZipFile(z);
e = zf.entries(); names = {};
while e.hasMoreElements()
    n = char(e.nextElement().getName());
    if endsWith(n, '.csv') && ~startsWith(n, '__MACOSX'), names{end+1} = n; end %#ok<AGROW>
end
zf.close();
assert(numel(names) == 1, '%s: expected one CSV.', z);
end

function h = sha(f)
%% SHA-256 of a file via the system tool.
[s, o] = system(sprintf('shasum -a 256 "%s"', f));
assert(s == 0, 'shasum failed on %s', f);
h = strtok(o);
end
