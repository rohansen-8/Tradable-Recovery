function exp3_cross_maturity(outRoot, B)
%% Experiment 3: weekly vs daily realized nonlinear option exposure on a common calendar.
%
%   exp3_cross_maturity()                 % writes results/economic_interpretation/eff_dim/exp3/
%   exp3_cross_maturity(outRoot, B)       % e.g. a scratch folder and a small B for a check run
%
% Inputs (read-only): primary OOS payoff components and window fits (weekly,
% daily), the full-sample eff_dim model states (configuration only), and
% data/<h>/df.csv (hashed for provenance). No SDF fits or solves.
% Common window: formation dates from the later OOS start to the earlier OOS
% end; each maturity keeps its own observations and holding periods.
% Regression per maturity: O = a + b*Delta + c*Delta^2, O = total - forward
% (the option strip), Bartlett-HAC SE with lag bartlett_hac_bandwidth(n).
% Joint inference on c_daily - c_weekly: paired calendar-month block bootstrap.
% Blocks of ell consecutive months (circular over the window's months) are
% drawn and every weekly and daily observation formed in a drawn month enters
% the draw, so both maturities share the same resampled calendar and keep
% their unequal counts. B draws, seed 2026112203, ell = 1 (primary), 2, 3.
% p-value: (1 + #{|d* - d| >= |d|}) / (B + 1); 95% percentile interval; share of d* <= 0.
% Specifications: raw (primary); raw excluding formation dates in March-April
% 2025; and a descriptive normalization O/sd(O) on z and z^2 with
% z = Delta / s, s = (VIX/100) * sqrt(holding_days/365) (VIX at formation).
% Ex-ante diagnostic: (1/2) h_mm^{(t)}(1, v_t) of each saved window fit at its
% formation-date VIX (the local Carr-Madan option loading; not equal to c).
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % repository root
cd(root); run('drivers/setpath.m');
addpath(fullfile(root,'pipelines','empirical_oos_attribution'));   % oos_sha256
if nargin < 1 || isempty(outRoot), outRoot = fullfile(results_root(),'economic_interpretation','eff_dim'); end
if nargin < 2 || isempty(B), B = 9999; end
outDir = fullfile(outRoot, 'exp3');
fOut = fullfile(outDir, 'exp3_cross_maturity.mat');
assert(~isfile(fOut), 'exp3:Exists', 'Refusing to overwrite %s', fOut);
seed = 2026112203; blocks = [1 2 3]; H = {'weekly','daily'};

%% Common calendar window
C = cell(1,2);
for i = 1:2
    C{i} = readtable(fullfile(results_root(),'empirical_oos','eff_dim','primary',H{i},[H{i} '_oos_components.csv']));
    C{i}.oos_date = datetime(C{i}.oos_date);
end
lo = max(min(C{1}.oos_date), min(C{2}.oos_date)); hi = min(max(C{1}.oos_date), max(C{2}.oos_date));
months = unique(dateshift([C{1}.oos_date(C{1}.oos_date >= lo & C{1}.oos_date <= hi); ...
                           C{2}.oos_date(C{2}.oos_date >= lo & C{2}.oos_date <= hi)], 'start', 'month'));
S = struct();
for i = 1:2
    T = C{i}(C{i}.oos_date >= lo & C{i}.oos_date <= hi, :);
    O = T.total - T.forward;
    cells = 0; for b = 1:5, cells = cells + T.(sprintf('c%d_put',b)) + T.(sprintf('c%d_call',b)); end
    assert(max(abs(O - cells)) < 1e-12, 'exp3:Identity', 'option strip != sum of cells');
    s = (T.vix / 100) .* sqrt(T.holding_days / 365);
    [~, mo] = ismember(dateshift(T.oos_date, 'start', 'month'), months);
    S.(H{i}) = struct('date', T.oos_date, 'window_id', T.window_id, 'O', O, 'D', T.Delta, ...
        'z', T.Delta ./ s, 'vix', T.vix, 'hold', T.holding_days, 'month', mo, 'pos', T.fwd_position);
end
excl = @(d) d >= datetime(2025,3,1) & d < datetime(2025,5,1);

%% Joint bootstrap self-check (before any inference)
chk = self_check(S, numel(months));

%% Specifications
specs = {'raw', 'raw_excl_MarApr2025', 'standardized'};
coef = table(); boot = table(); est = struct();
for k = 1:numel(specs)
    [Y, X, keepM] = build_spec(S, specs{k}, excl, numel(months));
    for i = 1:2
        n = numel(Y{i}); L = bartlett_hac_bandwidth(n);
        [bh, V] = ols_hac(Y{i}, X{i}, L); e = Y{i} - X{i}*bh;
        coef = [coef; table(string(specs{k}), string(H{i}), n, L, bh(1), bh(2), bh(3), sqrt(V(3,3)), bh(3)/sqrt(V(3,3)), ...
            1 - sum(e.^2)/sum((Y{i}-mean(Y{i})).^2), 'VariableNames', ...
            {'spec','horizon','n','hac_lag','a','b','c','se_c_hac','t_c_hac','R2'})]; %#ok<AGROW>
        est.(specs{k}).(H{i}) = bh(3);
    end
    d0 = est.(specs{k}).daily - est.(specs{k}).weekly;
    for ell = blocks
        ds = boot_diff(Y, X, keepM, ell, B, seed);
        ok = isfinite(ds); nv = sum(ok);
        p2 = (1 + sum(abs(ds(ok) - d0) >= abs(d0))) / (nv + 1);
        q = sort(ds(ok)); ci = q(min(nv, max(1, ceil([0.025 0.975] * (nv + 1)))));
        boot = [boot; table(string(specs{k}), ell, numel(unique(keepM{1})), d0, mean(ds(ok)), std(ds(ok)), ci(1), ci(2), p2, mean(ds(ok) <= 0), nv, ...
            'VariableNames', {'spec','block_months','n_months','diff_daily_minus_weekly','boot_mean','boot_se','ci95_lo','ci95_hi','p_two_sided','share_draws_le_0','n_valid'})]; %#ok<AGROW>
    end
end

%% Ex-ante local option loading (1/2) h_mm(1, v_t) from the saved window fits
ex = table(); exs = table();
for i = 1:2
    h = H{i}; L0 = load(fullfile(results_root(),'empirical','eff_dim',h,[h '_model_state.mat']));
    ms0 = L0; if isfield(L0,'model_state'), ms0 = L0.model_state; end
    wd = fullfile(results_root(),'empirical_oos','eff_dim','primary',h,'windows');
    files = dir(fullfile(wd,'w*.mat')); ids = cellfun(@(f) sscanf(f,'w%d_'), {files.name});
    G = S.(h); hh = nan(numel(G.O),1); pr = hh;
    for j = 1:numel(G.O)
        Wf = load(fullfile(wd, files(ids == G.window_id(j)).name)); W = Wf.W;
        assert(W.window.oos_date == G.date(j), 'exp3:Window', 'window/date mismatch');
        ms = ms0; ms.h_star = W.fit.h_star; ms.ls_endo = W.fit.ls_endo; ms.ls_exo = W.fit.ls_exo;
        ms.exo_grid = linspace(W.fit.exo_range(1), W.fit.exo_range(2), numel(ms0.exo_grid)).';
        ms.cfg.training.exo_domain = W.fit.exo_range; ms.cfg.training.exo_grid = ms.exo_grid;
        e = build_exo_representer(ms.cfg, ms, [], struct('mode','conditional','value',G.vix(j)));
        dd = evaluate_sdf_derivatives(ms.cfg, ms, 1, e);
        hh(j) = 0.5 * dd.h_mm(1); pr(j) = dd.h_m(1);
    end
    assert(max(abs(pr - G.pos)) < 1e-8, 'exp3:Rebuild', 'rebuilt forward position differs from Stage E');
    ex = [ex; table(repmat(string(h),numel(hh),1), G.date, G.vix, hh, G.O, G.D, 'VariableNames', ...
        {'horizon','oos_date','vix','half_hmm_atm','option_payoff','Delta'})]; %#ok<AGROW>
    exs = [exs; table(string(h), numel(hh), mean(hh), median(hh), std(hh), min(hh), max(hh), mean(hh > 0), ...
        corr(hh, G.vix), est.raw.(h), max(abs(pr - G.pos)), 'VariableNames', {'horizon','n','mean','median','sd','min','max', ...
        'share_positive','corr_with_vix','c_raw_realized','max_abs_pos_rebuild'})]; %#ok<AGROW>
end

%% Save with provenance
if ~isfolder(outDir), mkdir(outDir); end
[~, head] = system(sprintf('git -C "%s" rev-parse HEAD', root));
[~, dirty] = system(sprintf('git -C "%s" status --porcelain -- pipelines/economic_interpretation/exp3_cross_maturity.m', root));
meta = struct('created', string(datetime('now')), 'window', [lo hi], 'n_months', numel(months), ...
    'B', B, 'seed', seed, 'blocks', blocks, 'excluded', 'formation dates 2025-03-01..2025-04-30', ...
    'git_head', strtrim(head), 'driver_git_status', strtrim(dirty), ...
    'driver_sha256', oos_sha256(fileread([mfilename('fullpath') '.m'])), 'matlab_version', version);
for i = 1:2
    fc = fullfile(results_root(),'empirical_oos','eff_dim','primary',H{i},[H{i} '_oos_components.csv']);
    fd = fullfile(root,'data',H{i},'df.csv');
    meta.inputs.(H{i}) = struct('components', fc, 'components_sha256', sha_file(fc), 'panel', fd, 'panel_sha256', sha_file(fd), ...
        'windows', fullfile(results_root(),'empirical_oos','eff_dim','primary',H{i},'windows'));
end
smp = table(string(H'), [numel(S.weekly.O); numel(S.daily.O)], [min(S.weekly.date); min(S.daily.date)], ...
    [max(S.weekly.date); max(S.daily.date)], [sum(~excl(S.weekly.date)); sum(~excl(S.daily.date))], ...
    'VariableNames', {'horizon','n_common','first_formation','last_formation','n_excl_MarApr2025'});
save(fOut, 'meta', 'chk', 'smp', 'coef', 'boot', 'ex', 'exs', '-v7');
writetable(smp, fullfile(outDir,'exp3_sample.csv')); writetable(coef, fullfile(outDir,'exp3_coefficients.csv'));
writetable(boot, fullfile(outDir,'exp3_bootstrap.csv')); writetable(exs, fullfile(outDir,'exp3_exante_summary.csv'));
writetable(ex, fullfile(outDir,'exp3_exante_by_date.csv'));
disp(smp); disp(chk); disp(coef); disp(boot); disp(exs);
end

function [Y, X, keepM] = build_spec(S, spec, excl, M)
H = {'weekly','daily'}; Y = cell(1,2); X = Y; keepM = Y;
for i = 1:2
    G = S.(H{i}); keep = true(size(G.O));
    if strcmp(spec, 'raw_excl_MarApr2025'), keep = ~excl(G.date); end
    if strcmp(spec, 'standardized')
        Y{i} = G.O / std(G.O); x = G.z;
    else
        Y{i} = G.O(keep); x = G.D(keep);
    end
    X{i} = [ones(numel(x),1) x x.^2]; keepM{i} = G.month(keep);
end
% month indices restricted to months that remain (both maturities share them)
used = unique([keepM{1}; keepM{2}]); [~, keepM{1}] = ismember(keepM{1}, used); [~, keepM{2}] = ismember(keepM{2}, used);
assert(numel(used) <= M);
end

function ds = boot_diff(Y, X, mon, ell, B, seed)
%% Paired calendar-month block bootstrap of c_daily - c_weekly.
M = max([mon{1}; mon{2}]); nb = ceil(M / ell);
rows = cell(2, M);
for i = 1:2, for m = 1:M, rows{i,m} = find(mon{i} == m); end, end
rs = RandStream('twister', 'Seed', seed); ds = nan(B,1);
for r = 1:B
    st = randi(rs, M, 1, nb);
    ms = mod(st - 1 + (0:ell-1).', M) + 1; ms = ms(:); ms = ms(1:M);
    c = zeros(1,2);
    for i = 1:2
        ii = vertcat(rows{i, ms}); Xi = X{i}(ii,:);
        if rank(Xi) < 3, c(i) = NaN; else, b = Xi \ Y{i}(ii); c(i) = b(3); end
    end
    ds(r) = c(2) - c(1);
end
end

function chk = self_check(S, M)
%% Correctness checks of the joint resampler on the real data.
H = {'weekly','daily'}; Y = cell(1,2); X = Y; mon = Y; c0 = zeros(1,2);
for i = 1:2
    G = S.(H{i}); Y{i} = G.O; X{i} = [ones(numel(G.O),1) G.D G.D.^2]; mon{i} = G.month; b = X{i} \ Y{i}; c0(i) = b(3);
end
% (1) every month has observations of both maturities; month indices cover 1..M
chk.months = M; chk.all_months_both = all(ismember(1:M, mon{1})) && all(ismember(1:M, mon{2}));
% (2) one block of length M is a circular rotation of all months: estimate unchanged
ds = boot_diff(Y, X, mon, M, 5, 1);
chk.full_rotation_max_dev = max(abs(ds - (c0(2) - c0(1))));
% (3) a draw contains exactly the observations of its months, with repetition
rs = RandStream('twister', 'Seed', 7); st = randi(rs, M, 1, M); n = zeros(1,2);
for i = 1:2, n(i) = sum(arrayfun(@(m) sum(mon{i} == m), st)); end
cnt = cellfun(@(m) accumarray(m, 1, [M 1]), mon, 'UniformOutput', false);
chk.draw_counts_match = isequal(n, [sum(cnt{1}(st)) sum(cnt{2}(st))]);
chk.weekly_per_month_min = min(cnt{1}); chk.daily_per_month_min = min(cnt{2});
assert(chk.all_months_both && chk.full_rotation_max_dev < 1e-9 && chk.draw_counts_match, 'exp3:SelfCheck', 'bootstrap self-check failed');
end

function [b, V] = ols_hac(y, X, L)
%% OLS with Bartlett-HAC covariance (as summarize_oos_attribution / exp2_option_role).
n = numel(y); b = X \ y; e = y - X*b; Sxx = (X'*X)/n;
G = X .* e; S = (G'*G)/n;
for l = 1:L, Cl = (G(l+1:end,:)'*G(1:end-l,:))/n; S = S + (1 - l/(L+1))*(Cl + Cl'); end
V = (Sxx \ S) / Sxx / n;
end

function h = sha_file(f)
fid = fopen(f, 'r'); assert(fid > 0, 'exp3:Open', 'Cannot open %s', f); c = onCleanup(@() fclose(fid));
h = oos_sha256(fread(fid, Inf, '*uint8'));
end
