function outputs = render_simulation_recovery_boxplots(lambdaMethod)
%% Render the BS and CHJ SDF recovery-error boxplots from saved results.
%
% Both designs use one recovery measure, the HJ-normalized SDF
% (eq. hj_normalized_sdf), read from the files written by the refit drivers
% for the same T = recovery_T datasets and lambdas as the production fits:
%   BS:  bs_recovery_hj.mat  (run_bs_hj_recovery_refit)
%   CHJ: chj_recovery_hj.mat (run_chj_hj_recovery_refit)
% The recovery rows saved inside bs_results.mat / chj_results.mat depend on
% the level of h, which the payoffs do not identify and which mean-one
% normalization distorts when lambda is small, so they are not used. No
% simulation, fitting or inference is run here.
%
% lambdaMethod: 'eff_dim' (default, get_frozen_design().
% simulation_lambda_method) or 'cv'. Figures are written to
% simulation_output_dir('figures', lambdaMethod)/<experiment>/.
%
% Figure IA.A.1: bs_increasing_t/bs_sdf_recovery_boxplot.{png,pdf}
% Figure IA.A.2: chj_increasing_t/chj_sdf_recovery_boxplot.{png,pdf}

setpath();
if nargin < 1 || isempty(lambdaMethod)
    lambdaMethod = get_frozen_design().simulation_lambda_method;
end

simRoot = simulation_output_dir('simulations', lambdaMethod);
figRoot = simulation_output_dir('figures', lambdaMethod);
specs = { ...
    'BS',  'bs_increasing_t',  'bs_recovery_hj.mat', 'bs_sdf_recovery_boxplot', ...
           config_bs_simulation_design('monthly', 'bs_increasing_t', lambdaMethod), 'recovery_hj_raw'; ...
    'CHJ', 'chj_increasing_t', 'chj_recovery_hj.mat', 'chj_sdf_recovery_boxplot', ...
           config_chj_simulation_design(lambdaMethod), 'recovery_hj_raw'};
refitDriver = struct('BS', 'run_bs_hj_recovery_refit', 'CHJ', 'run_chj_hj_recovery_refit');

model = strings(0, 1); B = zeros(0, 1); K = zeros(0, 1);
pngFile = strings(0, 1); pdfFile = strings(0, 1);
for j = 1:size(specs, 1)
    resultFile = fullfile(simRoot, specs{j, 2}, specs{j, 3});
    assert(isfile(resultFile), ['Missing %s HJ recovery file: %s\n' ...
        'Run %s(''%s'') first.'], specs{j, 1}, resultFile, ...
        refitDriver.(specs{j, 1}), lambdaMethod);
    S = load(resultFile, specs{j, 6});
    Tval = specs{j, 5}.diagnostics.recovery_T;
    [m, diffs] = collect_sdf_recovery_differences(S.(specs{j, 6}), Tval);

    outBase = fullfile(figRoot, specs{j, 2}, specs{j, 4});
    fig = plot_sdf_recovery_boxplot(m, diffs, outBase);
    close(fig);

    model(end+1, 1) = string(specs{j, 1}); %#ok<AGROW>
    B(end+1, 1) = size(diffs, 1); %#ok<AGROW>
    K(end+1, 1) = numel(m); %#ok<AGROW>
    pngFile(end+1, 1) = string([outBase '.png']); %#ok<AGROW>
    pdfFile(end+1, 1) = string([outBase '.pdf']); %#ok<AGROW>
    fprintf('%s recovery boxplot (T=%d, %d datasets, %d grid points):\n  %s\n', ...
        specs{j, 1}, Tval, size(diffs, 1), numel(m), pngFile(end));
end

outputs = table(model, B, K, pngFile, pdfFile);

end

function [m, diffs, seeds] = collect_sdf_recovery_differences(recovery_raw, Tval)
%% Arrange saved production recovery rows as a datasets-by-grid matrix.
%
% recovery_raw is the recovery_hj_raw table of the HJ refit drivers (one row
% per dataset and grid point, column 'difference' = HJ-normalized estimated
% minus true SDF). Only rows with the requested Tval are used. Returns the sorted grid m (K-by-1), the B-by-K matrix diffs and
% the B-by-1 dataset seeds. No simulation or fitting is performed.

assert(istable(recovery_raw) && ~isempty(recovery_raw), ...
    'collect_sdf_recovery_differences:Empty', 'No saved recovery rows.');
need = {'Tval','seed','m','difference'};
assert(all(ismember(need, recovery_raw.Properties.VariableNames)), ...
    'collect_sdf_recovery_differences:Columns', ...
    'Recovery rows must contain Tval, seed, m and difference.');

R = recovery_raw(recovery_raw.Tval == Tval, :);
assert(~isempty(R), 'collect_sdf_recovery_differences:NoT', ...
    'No saved recovery rows at T = %d.', Tval);

seeds = unique(R.seed);
m = unique(R.m);
[~, iSeed] = ismember(R.seed, seeds);
[~, iM] = ismember(R.m, m);
assert(height(R) == numel(seeds) * numel(m), ...
    'collect_sdf_recovery_differences:Incomplete', ...
    'Recovery rows do not form a complete datasets-by-grid panel.');

diffs = NaN(numel(seeds), numel(m));
diffs(sub2ind(size(diffs), iSeed, iM)) = R.difference;
assert(all(isfinite(diffs(:))), 'collect_sdf_recovery_differences:Duplicate', ...
    'Recovery panel has missing or duplicated (seed, m) cells.');

end

function fig = plot_sdf_recovery_boxplot(m, diffs, outBase)
%% Plot estimated-minus-true SDF boxplots in the manuscript figure style.
%
%   m       : evaluation grid (forward moneyness), length K
%   diffs   : B-by-K pointwise recovery differences, one row per simulated
%             dataset (HJ-normalized estimated minus true SDF)
%   outBase : output path without extension; writes <outBase>.png and .pdf
%
% Produces the manuscript Figures IA.A.1-A.2: 6.0 x 3.5 in, serif type (axis labels 12 pt,
% tick labels 11 pt), inward ticks on the bottom and left axes only, up to
% 30 boxes in [0.95, 1.05], steelblue boxes (alpha 0.55) with a dark-red
% median, whiskers at 1.5 IQR, faint outlier dots, a black dashed zero
% line, and y-limits Q1 - 4 IQR to Q3 + 4 IQR padded by 10%. Quantiles use
% linear interpolation (Hyndman-Fan type 7). Boxes are drawn directly, so the
% Statistics toolbox is not required. Plotting only: no simulation or fitting.

m = m(:);
assert(size(diffs, 2) == numel(m), 'Recovery matrix/grid dimensions are inconsistent.');
assert(all(isfinite(diffs(:))), 'Recovery differences must be finite.');

% Up to 30 equally spaced grid indices (truncated to integers), then keep [0.95,1.05].
xLim = [0.95 1.05];
nBoxes = min(30, numel(m));
idx = floor(linspace(0, numel(m) - 1, nBoxes)) + 1;
mSel = m(idx);
mask = mSel >= xLim(1) & mSel <= xLim(2);
idx = idx(mask);
mSel = mSel(mask);
assert(numel(mSel) >= 2, 'Too few box locations in [0.95,1.05].');

nSel = numel(idx);
stats = zeros(nSel, 3);                     % q25, median, q75
for j = 1:nSel
    stats(j, :) = local_quantile7(diffs(:, idx(j)), [0.25 0.5 0.75]);
end
iqrVals = stats(:, 3) - stats(:, 1);
ylo = min(stats(:, 1) - 4 * iqrVals);
yhi = max(stats(:, 3) + 4 * iqrVals);
if yhi > ylo
    pad = 0.1 * (yhi - ylo);
else
    pad = max(1e-8, 0.1 * max(1, abs(yhi)));
end

steel = [70 130 180] / 255;                 % steel blue
accent = [139 26 26] / 255;                 % '#8b1a1a'
% Box width 0.6 x spacing; the production shape-test grid is
% not uniform (region-specific spacing), so use the smallest gap to keep
% adjacent boxes from overlapping.
boxW = 0.6 * min(diff(mSel));
capW = 0.5 * boxW;                          % whisker cap width

fig = figure('Color', 'w', 'Visible', 'off', 'Units', 'inches', ...
    'Position', [1 1 6.0 3.5], 'PaperUnits', 'inches', 'PaperSize', [6.0 3.5]);
ax = axes(fig);
hold(ax, 'on');

for j = 1:nSel
    x = mSel(j);
    d = diffs(:, idx(j));
    q1 = stats(j, 1); q2 = stats(j, 2); q3 = stats(j, 3);
    lowFence = q1 - 1.5 * iqrVals(j);
    highFence = q3 + 1.5 * iqrVals(j);
    wLo = min(d(d >= lowFence));
    wHi = max(d(d <= highFence));
    out = d(d < lowFence | d > highFence);

    if ~isempty(out)
        scatter(ax, repmat(x, numel(out), 1), out, 4, steel, 'filled', ...
            'MarkerFaceAlpha', 0.4, 'MarkerEdgeAlpha', 0.4);
    end
    plot(ax, [x x], [wLo q1], '-', 'Color', steel, 'LineWidth', 0.8);
    plot(ax, [x x], [q3 wHi], '-', 'Color', steel, 'LineWidth', 0.8);
    plot(ax, x + [-capW capW] / 2, [wLo wLo], '-', 'Color', steel, 'LineWidth', 0.8);
    plot(ax, x + [-capW capW] / 2, [wHi wHi], '-', 'Color', steel, 'LineWidth', 0.8);
    patch(ax, x + [-1 1 1 -1] * boxW / 2, [q1 q1 q3 q3], steel, ...
        'FaceAlpha', 0.55, 'EdgeColor', steel, 'LineWidth', 0.8);
    plot(ax, x + [-1 1] * boxW / 2, [q2 q2], '-', 'Color', accent, 'LineWidth', 1.5);
end
yline(ax, 0, '--', 'Color', [0 0 0], 'LineWidth', 1.0);

xt = linspace(xLim(1), xLim(2), 5);
set(ax, 'XLim', xLim, 'YLim', [ylo - pad, yhi + pad], ...
    'XTick', xt, 'XTickLabel', compose('%.2f', xt), ...
    'FontName', 'Times New Roman', 'FontSize', 11, ...
    'LineWidth', 0.6, 'TickDir', 'in', 'TickLength', [0.012 0.012], ...
    'Box', 'on', 'XMinorTick', 'off', 'YMinorTick', 'off', 'Layer', 'top');
ax.XAxis.TickLabelInterpreter = 'none';
xlabel(ax, 'Forward moneyness', 'FontName', 'Times New Roman', 'FontSize', 12);
ylabel(ax, 'Difference', 'FontName', 'Times New Roman', 'FontSize', 12);
ax.Toolbar.Visible = 'off';
hold(ax, 'off');

% Inward ticks on the bottom/left axes only (no top/right ticks):
% overlay a tick-free box so the frame is closed without top/right ticks.
axFrame = axes(fig, 'Position', ax.Position, 'Color', 'none', ...
    'XTick', [], 'YTick', [], 'Box', 'on', 'LineWidth', 0.6, ...
    'XColor', [0 0 0], 'YColor', [0 0 0], 'HitTest', 'off');
ax.Box = 'off';
linkprop([ax axFrame], 'Position');

outDir = fileparts(outBase);
if ~isempty(outDir) && ~isfolder(outDir), mkdir(outDir); end
exportgraphics(fig, [outBase '.png'], 'Resolution', 300);
exportgraphics(fig, [outBase '.pdf'], 'ContentType', 'vector');

end

function q = local_quantile7(x, p)
%% Linear-interpolation sample quantiles (Hyndman-Fan type 7).

x = sort(x(:));
n = numel(x);
h = (n - 1) * p(:) + 1;
lo = floor(h);
hi = min(lo + 1, n);
q = x(lo) + (h - lo) .* (x(hi) - x(lo));
q = q(:).';

end
