function T = build_fixed_lambda_tables(stage)
%% Fixed-lambda increasing-T publication tables from saved simulation outputs only.
%
%   build_fixed_lambda_tables('bs_increasing_t');  build_fixed_lambda_tables('chj_increasing_t');
%
% Reads results/simulations/fixed_lambda/<stage>/lambda_<tag>/<bs|chj>_rejection_summary.csv
% (5% Monte Carlo rejection frequencies, B replications per cell) for the pre-specified lambdas, and
% attaches the population shape classification of the regularized target h_lambda at the same
% lambda (deterministic: BS exact population map; CHJ saved 50,000-date population checkpoint,
% the source of Table IA.A.2). Nothing is simulated. Writes <stage>/tables/:
%   fixed_lambda_rejection_long.csv, fixed_lambda_rejection_wide.csv,
%   population_classification.csv, fixed_lambda_table.tex, fixed_lambda_table.md

setpath();
stage = validatestring(char(string(stage)), {'bs_increasing_t','chj_increasing_t'}, mfilename, 'stage');
root = fullfile(simulation_output_dir('simulations', 'fixed_lambda'), stage);
man = readtable(fullfile(root, 'fixed_lambda_design.csv'), 'TextType', 'string');
lambdas = man.lambda(:).';
prefix = extractBefore(stage, '_');
outDir = fullfile(root, 'tables'); if ~isfolder(outDir), mkdir(outDir); end

%% Saved rejection frequencies
rows = cell(numel(lambdas), 1);
for il = 1:numel(lambdas)
    f = fullfile(root, fixed_lambda_tag(lambdas(il)), [prefix '_rejection_summary.csv']);
    assert(isfile(f), 'build_fixed_lambda_tables:Missing', 'Missing saved results: %s', f);
    S = readtable(f, 'TextType', 'string');
    assert(all(S.lambda_mode == "fixed_lambda"), '%s is not a fixed_lambda run.', f);
    assert(all(S.lambda_fit_median == lambdas(il)) && all(S.lambda_inference_median == lambdas(il)), ...
        '%s does not carry the fixed lambda %g.', f, lambdas(il));
    rows{il} = table(repmat(lambdas(il), height(S), 1), S.Tval, S.region, S.testType, S.B, ...
        S.valid_tests, S.rej_5pct, S.se_5pct, ...
        'VariableNames', {'lambda', 'Tval', 'region', 'testType', 'B', 'valid_tests', 'rej_5pct', 'se_5pct'});
end
L = vertcat(rows{:});

%% Population classification of h_lambda at the same lambdas
P = local_population(stage, lambdas);
writetable(P, fullfile(outDir, 'population_classification.csv'));
[tf, loc] = ismember(compose('%.17g|', L.lambda) + L.region + "|" + L.testType, ...
    compose('%.17g|', P.lambda) + P.region + "|" + P.testType);
assert(all(tf), 'Population classification missing for some cells.');
L.population_holds = P.population_holds(loc);
L.population_min_theta = P.population_min_theta(loc);
L.economic_holds = P.economic_holds(loc);
L = sortrows(L, {'testType', 'region', 'lambda', 'Tval'}, {'ascend', 'ascend', 'descend', 'ascend'});
writetable(L, fullfile(outDir, 'fixed_lambda_rejection_long.csv'));

%% Wide table: rows restriction x region x lambda, columns T
Tgrid = unique(L.Tval).';
restr = ["mono", "conv", "conc"]; regs = ["left", "center", "right"];
W = table();
for r = restr
    for g = regs
        for lam = lambdas
            X = L(L.testType == r & L.region == g & L.lambda == lam, :);
            if isempty(X), continue; end
            row = table(r, g, lam, X.population_holds(1), X.economic_holds(1), ...
                'VariableNames', {'testType', 'region', 'lambda', 'population_holds', 'economic_holds'});
            for t = Tgrid
                v = X.rej_5pct(X.Tval == t); if isempty(v), v = NaN; end
                row.(sprintf('rej5_T%d', t)) = v;
            end
            W = [W; row]; %#ok<AGROW>
        end
    end
end
writetable(W, fullfile(outDir, 'fixed_lambda_rejection_wide.csv'));
local_write_tex(W, Tgrid, stage, unique(L.B), fullfile(outDir, 'fixed_lambda_table.tex'));
local_write_md(W, Tgrid, stage, unique(L.B), fullfile(outDir, 'fixed_lambda_table.md'));
T = struct('long', L, 'wide', W, 'population', P, 'output_dir', outDir);
fprintf('Fixed-lambda tables written to %s\n', outDir);
end

function P = local_population(stage, lambdas)
%% Deterministic population classification (no simulation) at the fixed lambdas.
if strcmp(stage, 'bs_increasing_t')
    n = numel(lambdas);
    req = table(repmat("bs_increasing_t", n, 1), repmat("monthly", n, 1), lambdas(:), ...
        repmat("left", n, 1), repmat("mono", n, 1), ...
        'VariableNames', {'experiment', 'horizon', 'lambda_target', 'region', 'testType'});
    M = build_bs_two_target_population_map(req);
    P = table(M.lambda_target, string(M.region), string(M.testType), ...
        double(M.regularized_target_null_holds), M.regularized_min_theta, ...
        double(M.economic_benchmark_null_holds), ...
        'VariableNames', {'lambda', 'region', 'testType', 'population_holds', ...
        'population_min_theta', 'economic_holds'});
else
    cfg = config_chj_simulation_design('cv');
    dd = fullfile(cfg.results.path, 'diagnostics', 'chj_population_T50000_burn5000_grid52');
    C = load(fullfile(dd, 'chj_population_checkpoint.mat'), 'Q_alpha', 'K_h', 'alpha_bar', ...
        'L_all', 'loadingBlocks', 'trueShape', 'population_checkpoint_version');
    assert(strcmp(C.population_checkpoint_version, 'chj_bias_aware_population_v1'));
    tau = cfg.testing.tau; cells = {};
    for lam = lambdas
        A = 0.5 * ((C.Q_alpha + lam * C.K_h) + (C.Q_alpha + lam * C.K_h).');
        [R, flag] = chol(A, 'lower');
        assert(flag == 0, 'CHJ population system not positive definite at lambda %g.', lam);
        th = C.L_all.' * (R.' \ (R \ C.alpha_bar));
        for ib = 1:height(C.loadingBlocks)
            b = C.loadingBlocks(ib, :); t = th(b.first_index:b.last_index);
            e = C.trueShape(string(C.trueShape.region) == string(b.region) & ...
                string(C.trueShape.testType) == string(b.testType), :);
            cells{end+1, 1} = table(lam, string(b.region), string(b.testType), double(all(t >= -tau)), ...
                min(t), double(e.min_shape_value >= -tau), 'VariableNames', {'lambda', 'region', ...
                'testType', 'population_holds', 'population_min_theta', 'economic_holds'}); %#ok<AGROW>
        end
    end
    P = vertcat(cells{:});
end
P = sortrows(P, {'testType', 'region', 'lambda'}, {'ascend', 'ascend', 'descend'});
end

function local_write_tex(W, Tgrid, stage, B, file)
%% LaTeX table: panels by restriction, rows region x lambda, columns T.
names = struct('mono', 'Monotonicity (decreasing SDF)', 'conv', 'Convexity', 'conc', 'Concavity');
regn = containers.Map({'left', 'center', 'right'}, {'OTM put', 'Near ATM', 'OTM call'});
econ = containers.Map({'bs_increasing_t', 'chj_increasing_t'}, {'Black--Scholes', 'CHJ'});
fid = fopen(file, 'w'); c = onCleanup(@() fclose(fid));
fprintf(fid, '%% Generated by build_fixed_lambda_tables.m from saved simulation outputs. Do not edit.\n');
fprintf(fid, '\\begin{table}[!ht]\\centering\\small\n');
fprintf(fid, '\\caption{Fixed-$\\lambda$ shape-test rejection frequencies at the 5\\%% level, %s economy}\n', econ(stage));
fprintf(fid, '\\begin{tabular}{llc%s}\\toprule\n', repmat('r', 1, numel(Tgrid)));
fprintf(fid, 'Region & $\\lambda$ & $h_\\lambda$ & %s \\\\\n', strjoin(compose('$T=%d$', Tgrid), ' & '));
letters = 'ABC'; k = 0;
for r = ["mono", "conv", "conc"]
    X = W(W.testType == r, :); if isempty(X), continue; end
    k = k + 1;
    fprintf(fid, '\\midrule\\multicolumn{%d}{l}{\\textit{Panel %s: %s}}\\\\\n', 3 + numel(Tgrid), letters(k), names.(r));
    regsHere = unique(X.region, 'stable');
    for g = regsHere.'
        Y = X(X.region == g, :);
        for i = 1:height(Y)
            lab = ''; if i == 1, lab = regn(char(g)); end
            mark = '$\times$'; if Y.population_holds(i) == 1, mark = '$\checkmark$'; end
            vals = arrayfun(@(t) sprintf('%.1f', 100 * Y.(sprintf('rej5_T%d', t))(i)), Tgrid, 'UniformOutput', false);
            fprintf(fid, '%s & $10^{%d}$ & %s & %s \\\\\n', lab, round(log10(Y.lambda(i))), mark, strjoin(vals, ' & '));
        end
    end
end
fprintf(fid, '\\bottomrule\\end{tabular}\n');
fprintf(fid, ['\\par\\footnotesize\\textit{Notes:} Entries are Monte Carlo rejection frequencies (in %%) of the ' ...
    'regional one-sided Wald shape test at the 5\\%% nominal level across $B=%d$ independent samples, with ' ...
    '$\\lambda$ fixed across all $T$ and used for both estimation and inference. The column $h_\\lambda$ ' ...
    'reports whether the restriction holds ($\\checkmark$) or fails ($\\times$) for the regularized population ' ...
    'trading rule at that $\\lambda$; rejection frequencies are not size or power when the regularized target ' ...
    'differs from the economic benchmark.\n'], B);
fprintf(fid, '\\end{table}\n');
end

function local_write_md(W, Tgrid, stage, B, file)
%% Markdown preview of the same table.
fid = fopen(file, 'w'); c = onCleanup(@() fclose(fid));
fprintf(fid, '# %s fixed-lambda rejection frequencies (5%%, B = %d)\n\n', stage, B);
fprintf(fid, '| Restriction | Region | lambda | h_lambda | %s |\n', strjoin(compose('T=%d', Tgrid), ' | '));
fprintf(fid, '|%s\n', repmat('---|', 1, 4 + numel(Tgrid)));
for i = 1:height(W)
    mark = 'x'; if W.population_holds(i) == 1, mark = 'holds'; end
    vals = arrayfun(@(t) sprintf('%.1f', 100 * W.(sprintf('rej5_T%d', t))(i)), Tgrid, 'UniformOutput', false);
    fprintf(fid, '| %s | %s | %g | %s | %s |\n', W.testType(i), W.region(i), W.lambda(i), mark, strjoin(vals, ' | '));
end
end
