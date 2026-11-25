function exp2_option_role(h, outRoot)
%% Experiment 2: economic role of the option strip in the OOS eff_dim strategy.
% Inputs (read-only): primary OOS payoff components (gross payoffs; total =
% forward + 10 strike cells) and the OOS short variance-swap benchmark series
% (data/return_pairs; the series used in the Ledoit-Wolf test). No refits.
% Regions as in the OOS attribution: OTM put b2 [a,c1), near-ATM b3 [c1,c2), OTM call b4
% [c2,z], outside test domain b1+b5; put and call cells of each bucket summed.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % repository root
cd(root); run('drivers/setpath.m');
if nargin < 2 || isempty(outRoot), outRoot = fullfile(results_root(),'economic_interpretation','eff_dim'); end
outDir = fullfile(outRoot,'exp2',h);   % outRoot: default results/economic_interpretation/eff_dim
if ~isfolder(outDir), mkdir(outDir); end
fOut = fullfile(outDir, sprintf('%s_exp2.mat', h));
assert(~isfile(fOut), 'exp2:Exists', 'Refusing to overwrite %s', fOut);
O = readtable(fullfile(results_root(), sprintf('empirical_oos/eff_dim/primary/%s/%s_oos_components.csv',h,h)));
n = height(O); D = O.Delta; F = O.forward; R = O.total;
cellsum = @(b) O.(sprintf('c%d_put',b)) + O.(sprintf('c%d_call',b));
reg = struct('otm_put', cellsum(2), 'near_atm', cellsum(3), 'otm_call', cellsum(4), ...
             'outside', cellsum(1) + cellsum(5));
Opt = reg.otm_put + reg.near_atm + reg.otm_call + reg.outside;
% ---- accounting identities (every date)
A.max_total_identity = max(abs(R - (F + Opt)));
A.max_identity123 = max(abs([O.identity1; O.identity2; O.identity3]));
assert(A.max_total_identity < 1e-12, 'exp2:Identity', 'components do not sum to total');
hasVS = ~strcmp(h, 'quarterly');
if hasVS
    P = readtable(fullfile(root,'data','return_pairs',sprintf('oos_%s_return_pairs.csv', h)));
    dP = P.date; if ~isdatetime(dP), dP = datetime(string(dP), 'InputFormat', 'dd-MMM-yyyy', 'Locale', 'en_US'); end
    dO = O.oos_date; if ~isdatetime(dO), dO = datetime(string(dO), 'InputFormat', 'yyyy-MM-dd'); end
    assert(height(P) == n && all(dateshift(dP,'start','day') == dateshift(dO,'start','day')), ...
        'exp2:VS', 'VS series misaligned');
    VS = P.vswap_return;
end
k = ceil(0.10 * n);                       % size of every 10% tail
ann = struct('quarterly',4,'monthly',12,'weekly',52,'daily',252); N = ann.(h);
% ---- A: downside statistics (point estimates)
[st, idx] = downside_stats(D, F, R, Opt, k);
% ---- B: mean / variance / tail attribution (exact)
cp = @(x,y) mean((x-mean(x)).*(y-mean(y)));
names = fieldnames(reg);
B = struct('mean_forward_ann', mean(F)*N, 'mean_options_ann', mean(Opt)*N, 'mean_total_ann', mean(R)*N, ...
    'var_forward', cp(F,F), 'var_options', cp(Opt,Opt), 'two_cov_fo', 2*cp(F,Opt), 'var_total', cp(R,R), ...
    'corr_fo', cp(F,Opt)/sqrt(cp(F,F)*cp(Opt,Opt)), 'sr_forward', mean(F)/std(F)*sqrt(N), ...
    'sr_total', mean(R)/std(R)*sqrt(N), 'sr_options', mean(Opt)/std(Opt)*sqrt(N));
B.resid_var = abs(B.var_total - (B.var_forward + B.var_options + B.two_cov_fo));
wt = idx.worst_total;                     % total's own worst-k dates
B.es_total_from_forward = mean(F(wt)); B.es_total_from_options = mean(Opt(wt));
regTab = table('Size',[numel(names)+1 6], 'VariableTypes', repmat({'double'},1,6), ...
    'VariableNames', {'mean_ann','euler_var_share','cov_with_forward','mean_mkt_worst','mean_fwd_worst','es_total_contrib'}, ...
    'RowNames', [{'forward'}; names]);
comps = [{F}; cellfun(@(f) reg.(f), names, 'UniformOutput', false)];
for i = 1:numel(comps)
    x = comps{i};
    regTab{i,:} = [mean(x)*N, cp(x,R)/cp(R,R), cp(x,F), mean(x(idx.worst_mkt)), mean(x(idx.worst_fwd)), mean(x(wt))];
end
% ---- paired circular-block bootstrap of the downside differences
boot = struct();
if hasVS
    L = bartlett_hac_bandwidth(n); b0 = max(1, L);
    Bn = 9999; seed = 2026112201; blocks = unique([1 b0 2*b0]);
    for j = 1:numel(blocks)
        rs = RandStream('twister','Seed',seed);
        th = zeros(Bn, 6);
        nb = ceil(n / blocks(j));
        for c0 = 1:1000:Bn
            cc = c0:min(Bn, c0+999); m = numel(cc);
            S0 = randi(rs, n, nb, m);
            I = mod(reshape(S0,1,nb,m) - 1 + (0:blocks(j)-1).', n) + 1;
            I = reshape(I, blocks(j)*nb, m); I = I(1:n, :);
            for q = 1:m
                ii = I(:,q); s = downside_stats(D(ii), F(ii), R(ii), Opt(ii), k);
                th(cc(q),:) = stat_vec(s);
            end
        end
        t0 = stat_vec(st);
        p2 = (1 + sum(abs(th - t0) >= abs(t0), 1)) / (Bn + 1);
        lo = local_q(th, 0.025); hi = local_q(th, 0.975); se = std(th, 0, 1);
        boot.(sprintf('b%d', blocks(j))) = table(stat_names().', t0.', se.', lo.', hi.', p2.', ...
            'VariableNames', {'statistic','estimate','boot_se','ci95_lo','ci95_hi','p_two_sided'});
    end
    boot.primary_block = b0; boot.B = Bn; boot.seed = seed;
end
% ---- C: HAC exposure regressions (Bartlett, lag rule bartlett_hac_bandwidth; as in summarize_oos_attribution)
C = table();
if hasVS
    L = bartlett_hac_bandwidth(n);
    ys = [{'options', Opt}; [names, cellfun(@(f) reg.(f), names, 'UniformOutput', false)]];
    Dm = min(D,0); Dp = max(D,0);
    for i = 1:size(ys,1)
        y = ys{i,2};
        [b1,V1] = ols_hac(y, [ones(n,1) Dm Dp], L);
        [b2,V2] = ols_hac(y, [ones(n,1) D D.^2], L);
        [b3,V3] = ols_hac(y, [ones(n,1) VS], L);
        g = [0 1 -1]; tdiff = (g*b1)/sqrt(g*V1*g');
        C = [C; table(string(ys{i,1}), b1(2), b1(2)/sqrt(V1(2,2)), b1(3), b1(3)/sqrt(V1(3,3)), tdiff, r2(y,[ones(n,1) Dm Dp]), ...
            b2(2), b2(2)/sqrt(V2(2,2)), b2(3), b2(3)/sqrt(V2(3,3)), r2(y,[ones(n,1) D D.^2]), ...
            b3(1)*N, b3(1)/sqrt(V3(1,1)), b3(2), b3(2)/sqrt(V3(2,2)), r2(y,[ones(n,1) VS]), ...
            'VariableNames', {'series','b_down','t_down','b_up','t_up','t_down_minus_up','R2_S1', ...
            'b_mkt','t_mkt','b_sq','t_sq','R2_S2','alpha_vs_ann','t_alpha_vs','b_vs','t_vs','R2_S3'})]; %#ok<AGROW>
    end
    C.hac_lag = repmat(L, height(C), 1);
end
meta = struct('horizon',h,'n',n,'k_tail',k,'created',string(datetime('now')), ...
    'inputs', {{'results/empirical_oos/eff_dim/primary/<h>/<h>_oos_components.csv', ...
                'data/return_pairs/oos_<h>_return_pairs.csv'}});
save(fOut, 'A','st','B','regTab','boot','C','meta', '-v7');
writetable(struct2table(st), fullfile(outDir, sprintf('%s_exp2_downside.csv',h)));
writetable(struct2table(B), fullfile(outDir, sprintf('%s_exp2_attribution.csv',h)));
writetable(regTab, fullfile(outDir, sprintf('%s_exp2_regions.csv',h)), 'WriteRowNames', true);
if hasVS
    writetable(C, fullfile(outDir, sprintf('%s_exp2_exposures.csv',h)));
    fb = fieldnames(boot); fb = fb(startsWith(fb,'b') & ~strcmp(fb,'B'));
    for j = 1:numel(fb), writetable(boot.(fb{j}), fullfile(outDir, sprintf('%s_exp2_boot_%s.csv',h,fb{j}))); end
end
fprintf('== %s n=%d k=%d  identity %.1e\n', h, n, k, A.max_total_identity);
disp(struct2table(st)); disp(struct2table(B)); disp(regTab);
if hasVS, disp(C); for j = 1:numel(fb), fprintf('-- bootstrap %s\n', fb{j}); disp(boot.(fb{j})); end, end
end

function [s, idx] = downside_stats(D, F, R, Opt, k)
[~, o] = sort(D);  wm = o(1:k);           % worst market states
[~, o] = sort(F);  wf = o(1:k);           % forward's own worst dates
[~, o] = sort(R);  wt = o(1:k);           % total's own worst dates
s.mkt_worst_mean_forward = mean(F(wm)); s.mkt_worst_mean_total = mean(R(wm)); s.mkt_worst_mean_options = mean(Opt(wm));
s.mkt_worst_lossfreq_forward = mean(F(wm) < 0); s.mkt_worst_lossfreq_total = mean(R(wm) < 0);
s.lossfreq_forward = mean(F < 0); s.lossfreq_total = mean(R < 0);
s.es10_forward = mean(F(wf)); s.es10_total = mean(R(wt));
s.fwd_worst_mean_options = mean(Opt(wf)); s.fwd_worst_mean_total = mean(R(wf));
s.d_es10 = s.es10_total - s.es10_forward;
s.d_lossfreq = s.lossfreq_total - s.lossfreq_forward;
s.d_mkt_worst_lossfreq = s.mkt_worst_lossfreq_total - s.mkt_worst_lossfreq_forward;
s.var_total_minus_forward = var(R,1) - var(F,1);
idx = struct('worst_mkt', wm, 'worst_fwd', wf, 'worst_total', wt);
end

function v = stat_vec(s)
v = [s.d_es10, s.mkt_worst_mean_options, s.fwd_worst_mean_options, s.d_lossfreq, s.d_mkt_worst_lossfreq, s.var_total_minus_forward];
end
function c = stat_names()
c = {'ES10_total_minus_forward','options_mean_in_market_worst10','options_mean_in_forward_worst10', ...
     'lossfreq_total_minus_forward','lossfreq_diff_in_market_worst10','var_total_minus_forward'};
end
function q = local_q(X, p)
X = sort(X, 1); n = size(X,1); q = X(min(n, max(1, ceil(p*(n+1)))), :);
end
function [b, V] = ols_hac(y, X, L)
%% OLS with Bartlett-HAC covariance (as in summarize_oos_attribution).
n = numel(y); b = X \ y; e = y - X*b; Sxx = (X'*X)/n;
G = X .* e; S = (G'*G)/n;
for l = 1:L, Cl = (G(l+1:end,:)'*G(1:end-l,:))/n; S = S + (1 - l/(L+1))*(Cl + Cl'); end
V = (Sxx \ S) / Sxx / n;
end
function v = r2(y, X)
e = y - X*(X\y); v = 1 - sum(e.^2)/sum((y-mean(y)).^2);
end
