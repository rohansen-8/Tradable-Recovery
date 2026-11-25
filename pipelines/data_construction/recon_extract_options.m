function info = recon_extract_options(maxRows)
%% One chunked pass over the 48M-row OptionMetrics file: cleaning.py row filters and maturity pre-filter.
%
% info = recon_extract_options()          full file  -> extracted/options_candidates.mat
% info = recon_extract_options(maxRows)   row-limited test -> extracted/smoke/options_candidates.mat
%
% Row filters exactly as coded in cleaning.py (all row-wise, so order does
% not change the result; counts are reported sequentially in that order):
%   date in [2000-01-01, 2025-12-31]; strike/1000; mat = exdate - date (days);
%   df["bid"] > 0 | (df["ask"] > 0)   -> parses as bid > (ask > 0), i.e. bid > 1;
%   bid <= ask; open interest > 0; volume > 0; implied volatility present.
% Pre-filter mat in {1..9, 25..33, 80..100}: every later selection lies in
% these windows and the duplicate key (date, strike, mat, flag) contains mat.
% The VIX/forward filters and the keep=False de-duplication need merged
% data and are applied in recon_build_panels.
%
% For validation, the raw rows of every (date, optionid) in the distributed
% panels are also kept, before any filter. Refuses to overwrite.

c = recon_config();
smoke = nargin > 0 && ~isempty(maxRows);
if ~smoke, maxRows = inf; end
outDir = c.extracted; if smoke, outDir = fullfile(c.extracted, 'smoke'); end
outFile = fullfile(outDir, 'options_candidates.mat');
assert(~isfile(outFile), '%s exists; refusing to overwrite.', outFile);
if ~isfolder(outDir), mkdir(outDir); end

zipFile = fullfile(c.raw, c.zip.options);
csv = fullfile(c.extracted, 'tepvpvyv8tuiy2ve.csv');
if ~isfile(csv)
    [s, o] = system(sprintf('unzip -o -j "%s" "*.csv" -x "__MACOSX/*" -d "%s"', zipFile, c.extracted));
    assert(s == 0, 'unzip failed: %s', o);
end
assert(isfile(csv), 'Extracted options CSV not found.');

% Keys of the distributed panels (date, optionid) for raw-row capture
pk = [];
for h = 1:4
    o = detectImportOptions(fullfile(c.panels_existing, c.horizons{h}, 'df.csv'));
    o.SelectedVariableNames = {'date', 'optionid'};
    o = setvartype(o, 'date', 'datetime'); o = setvaropts(o, 'date', 'InputFormat', 'yyyy-MM-dd');
    P = readtable(fullfile(c.panels_existing, c.horizons{h}, 'df.csv'), o);
    pk = [pk; key(P.date, P.optionid)]; %#ok<AGROW>
end
pk = unique(pk);

vars = {'date', 'exdate', 'cp_flag', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', ...
    'impl_volatility', 'delta', 'gamma', 'vega', 'theta', 'optionid', 'am_settlement', 'contract_size'};
ds = tabularTextDatastore(csv, 'SelectedVariableNames', vars, 'ReadSize', 1e6);
ds.SelectedFormats = [{'%{yyyy-MM-dd}D', '%{yyyy-MM-dd}D', '%C'}, repmat({'%f'}, 1, 13)];

steps = {'read', 'date_range', 'bid_gt_(ask>0)', 'bid_le_ask', 'oi_gt_0', 'volume_gt_0', 'iv_present', 'mat_prefilter'};
cnt = zeros(1, numel(steps));
C = {}; Rk = {}; nRead = 0; lastDate = NaT; sorted = true;
t0 = tic;
while hasdata(ds) && nRead < maxRows
    T = read(ds);
    if nRead + height(T) > maxRows, T = T(1:maxRows - nRead, :); end
    nRead = nRead + height(T);
    sorted = sorted && issorted(T.date) && (isnat(lastDate) || T.date(1) >= lastDate);
    lastDate = T.date(end);
    T.mat = days(T.exdate - T.date);
    T.flag = double(T.cp_flag == 'C');          % C -> 1, P -> 0 (cleaning.py)
    T.cp_flag = [];
    % raw rows of the distributed panels' contracts, unfiltered
    hit = ismember(key(T.date, T.optionid), pk);
    if any(hit), Rk{end+1} = T(hit, :); end %#ok<AGROW>
    k = true(height(T), 1);
    m = {T.date >= c.clean_dates(1) & T.date <= c.clean_dates(2), ...
        T.best_bid > double(T.best_offer > 0), T.best_bid <= T.best_offer, T.open_interest > 0, ...
        T.volume > 0, ~isnan(T.impl_volatility), ismember(T.mat, c.mat_prefilter)};
    cnt(1) = cnt(1) + height(T);
    for j = 1:numel(m)
        k = k & m{j};
        cnt(j + 1) = cnt(j + 1) + sum(k);
    end
    C{end+1} = T(k, :); %#ok<AGROW>
    fprintf('%10d rows read, %8d kept, %s, %.0f s\n', nRead, sum(cellfun(@height, C)), string(lastDate), toc(t0));
end
C = vertcat(C{:});
C.strike = C.strike_price / 1000; C.strike_price = [];
R = vertcat(Rk{:});
R.strike = R.strike_price / 1000; R.strike_price = [];

info = struct();
info.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
info.matlab = version;
info.smoke = smoke; info.max_rows = maxRows; info.rows_read = nRead;
info.file_sorted_by_date = sorted;
info.filter_counts = cell2struct(num2cell(cnt), matlab.lang.makeValidName(steps), 2);
info.n_candidates = height(C);
info.n_panel_key_raw_rows = height(R);
info.n_panel_keys = numel(pk);
info.seconds = toc(t0);
info.code_sha256 = recon_sha(which('recon_extract_options'));
info.config_sha256 = recon_sha(which('recon_config'));
info.input_zip_sha256 = recon_sha(zipFile);
info.input_csv = csv;
save(outFile, 'C', 'R', 'info', '-v7.3');
fid = fopen(fullfile(outDir, 'extract_meta.json'), 'w');
fprintf(fid, '%s', jsonencode(info, 'PrettyPrint', true)); fclose(fid);
end

function k = key(d, id)
%% Exact scalar key for (date, optionid): datenum * 1e9 + optionid (< 2^53).
assert(all(id >= 0 & id < 1e9 & id == round(id)), 'optionid outside the key range.');
k = floor(datenum(d)) * 1e9 + id; %#ok<DATNM>
end
