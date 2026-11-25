function exp1_stageA(h, outRoot)
%% Experiment 1, step 1: no-look-ahead constant forward exposure c_t and accounting.
% c_t = mean over training dates s of h_m^{(t)}(1, v_s), window fit h^{(t)} rebuilt
% from the saved eff_dim window (h_star, length scales, VIX range). Read-only on inputs.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % repository root
cd(root); run('drivers/setpath.m');
if nargin < 2 || isempty(outRoot), outRoot = fullfile(results_root(),'economic_interpretation','eff_dim'); end
outDir = fullfile(outRoot,'exp1',h);   % outRoot: default results/economic_interpretation/eff_dim
if ~isfolder(outDir), mkdir(outDir); end
fA = fullfile(outDir, sprintf('%s_exp1_accounting.csv', h));
assert(~isfile(fA), 'exp1:Exists', 'Refusing to overwrite %s', fA);
ann = struct('quarterly',4,'monthly',12,'weekly',52,'daily',252); A = ann.(h);
L = load(fullfile(results_root(), sprintf('empirical/eff_dim/%s/%s_model_state.mat',h,h)));
ms0 = L; if isfield(L,'model_state'), ms0 = L.model_state; end
df = readtable(sprintf('data/%s/df.csv',h), 'VariableNamingRule','modify'); df.date = datetime(df.date);
[g, ud] = findgroups(df.date); n = splitapply(@numel, df.VIX, g); vx = splitapply(@(x) x(1), df.VIX, g);
keep = n >= 10; ud = ud(keep); vx = vx(keep);
oc = readtable(fullfile(results_root(), sprintf('empirical_oos/eff_dim/primary/%s/%s_oos_components.csv',h,h)));
oc.oos_date = datetime(oc.oos_date);
addpath(fullfile(root,'pipelines','empirical_oos_attribution'));
oosP = config_empirical_oos(h, struct()); panelP = load_oos_panel(oosP); sched = build_oos_schedule(oosP, panelP);
assert(numel(sched.windows) == height(oc), 'exp1:Sched', 'schedule size');
files = dir(fullfile(results_root(), sprintf('empirical_oos/eff_dim/primary/%s/windows/w*.mat',h)));
assert(numel(files) == height(oc), 'exp1:Count', 'window count');
N = height(oc); c = nan(N,1); posRe = nan(N,1); nNeg = nan(N,1); ntr = nan(N,1); sdTr = nan(N,1);
for k = 1:N
    W = load(fullfile(files(k).folder, files(k).name)); W = W.W;
    assert(W.window.id == oc.window_id(k) && W.window.oos_date == oc.oos_date(k), 'exp1:Align', 'window %d', k);
    ms = ms0; ms.h_star = W.fit.h_star; ms.ls_endo = W.fit.ls_endo; ms.ls_exo = W.fit.ls_exo;
    ms.exo_grid = linspace(W.fit.exo_range(1), W.fit.exo_range(2), numel(ms0.exo_grid)).';
    ms.cfg.training.exo_domain = W.fit.exo_range; ms.cfg.training.exo_grid = ms.exo_grid;
    sw = sched.windows(k); trd = panelP.dates(sw.train_index);
    assert(sw.oos_date == W.window.oos_date && sw.n_train == W.window.n_train && max(trd) == W.window.train_last, 'exp1:Train', 'window %d training dates', k);
    [tf, loc] = ismember(trd, ud); assert(all(tf), 'exp1:Dates', 'training date missing');
    vt = vx(loc); hm = nan(size(vt));
    for j = 1:numel(vt)
        e = build_exo_representer(ms.cfg, ms, [], struct('mode','conditional','value',vt(j)));
        d = evaluate_sdf_derivatives(ms.cfg, ms, 1, e); hm(j) = d.h_m(1);
    end
    e = build_exo_representer(ms.cfg, ms, [], struct('mode','conditional','value',W.state.vix));
    d = evaluate_sdf_derivatives(ms.cfg, ms, 1, e); posRe(k) = d.h_m(1);
    c(k) = mean(hm); nNeg(k) = sum(hm <= 0); ntr(k) = numel(vt); sdTr(k) = std(hm);
end
pos = oc.fwd_position; D = oc.Delta; F = oc.forward;
C = c .* D; tau = (pos - c) .* D;
T = table(oc.window_id, oc.oos_date, oc.vix, D, pos, posRe, c, sdTr, nNeg, ntr, F, C, tau, ...
    'VariableNames', {'window_id','oos_date','vix','Delta','fwd_position','fwd_position_rebuilt', ...
    'c_t','sd_train_position','n_train_nonpositive','n_train','forward','const_payoff','timing_payoff'});
writetable(T, fA);
% Accounting checks (every date)
chk.max_rebuild = max(abs(posRe - pos)); chk.max_fwd_pos_delta = max(abs(F - pos .* D));
chk.max_identity = max(abs(F - (C + tau)));
% Moments (population covariance, 1/N, so the decomposition is exact)
x = pos - c; mx = mean(x); mD = mean(D);
covp = @(a,b) mean((a-mean(a)).*(b-mean(b)));
S = struct('horizon',h,'N',N,'annualization',A, ...
  'sr_forward',mean(F)/std(F)*sqrt(A),'sr_const',mean(C)/std(C)*sqrt(A),'sr_timing',mean(tau)/std(tau)*sqrt(A), ...
  'sr_market',mean(D)/std(D)*sqrt(A), ...
  'mean_forward_ann',mean(F)*A,'mean_const_ann',mean(C)*A,'mean_timing_ann',mean(tau)*A, ...
  'timing_level_part_ann',mx*mD*A,'timing_cov_part_ann',covp(x,D)*A, ...
  'cov_timing_const_ann',covp(tau,C)*A,'corr_timing_const',corr(tau,C), ...
  'var_forward_ann',var(F,1)*A,'var_const_ann',var(C,1)*A,'var_timing_ann',var(tau,1)*A, ...
  'mean_pos',mean(pos),'mean_c',mean(c),'min_c',min(c),'max_c',max(c),'n_c_nonpositive',sum(c<=0), ...
  'n_pos_nonpositive',sum(pos<=0),'mean_dev',mx,'corr_dev_vix',corr(x,oc.vix), ...
  'max_abs_rebuild',chk.max_rebuild,'max_abs_fwd_pos_delta',chk.max_fwd_pos_delta,'max_abs_identity',chk.max_identity, ...
  'resid_mean_decomp',abs(mean(F) - (mean(C) + mx*mD + covp(x,D))), ...
  'resid_var_decomp',abs(var(F,1) - (var(C,1)+var(tau,1)+2*covp(tau,C))));
assert(S.max_abs_rebuild < 1e-8 && S.max_abs_identity < 1e-12, 'exp1:Identity', 'identity failed');
save(fullfile(outDir, sprintf('%s_exp1_summary.mat',h)), 'S');
writetable(struct2table(S), fullfile(outDir, sprintf('%s_exp1_summary.csv',h)));
disp(struct2table(S, 'AsArray', true));
end
