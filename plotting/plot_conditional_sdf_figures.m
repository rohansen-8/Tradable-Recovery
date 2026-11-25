function outputs = plot_conditional_sdf_figures(horizon, outDir, lambdaMethod)
%% Plot the conditional SDF figure from one saved fitted state.
%
% Reads the saved empirical fitted state, selects the five conditional VIX
% states with select_vix_levels, and evaluates the fitted function on
% m_raw = exp(linspace(figure_log_range, 300)). The five conditional curves
% are scaled by one common mean and drawn against log forward moneyness,
% with integer legend labels. The plotted curves are used for the figure
% only and never by the shape tests.

setpath();
horizon = validatestring(char(string(horizon)), ...
    {'quarterly','monthly','weekly','daily'}, mfilename, 'horizon');

design = get_frozen_design();
maturity = get_empirical_maturity_config(horizon);
if nargin < 3, lambdaMethod = []; end
cfg = config_empirical(horizon, lambdaMethod);
if nargin < 2 || isempty(outDir)
    outDir = cfg.results.figuresPath;
end
if ~isfolder(outDir), mkdir(outDir); end
assert(isfile(cfg.dataset.modelStateFile), ...
    'Missing saved fitted state: %s', cfg.dataset.modelStateFile);
S = load(cfg.dataset.modelStateFile, 'model_state');
ms = S.model_state;

[levels, selInfo] = select_vix_levels(cfg, ms);

nPts = design.figure_plot_points;
logRange = maturity.figure_log_range;
mLog = (0:nPts-1) * ((logRange(2)-logRange(1))/(nPts-1)) + logRange(1);
mLog(end) = logRange(2);
mRaw = exp(mLog(:));

curves = zeros(nPts, numel(levels));
for c = 1:numel(levels)
    exo = build_exo_representer(cfg, ms, [], ...
        struct('mode','conditional','value',levels(c)));
    out = evaluate_sdf_grid(cfg, ms, mRaw, exo);
    curves(:, c) = out.M(:);       % plotted function-value curve
end
mu = mean(curves(:));              % one common scaling across all curves
assert(isfinite(mu) && mu > 0, 'Common scaling mean must be positive.');
scaled = curves / mu;

f = figure('Visible','off','Position',[100 100 900 620]);
hold on; box on; grid on;
co = lines(numel(levels));
for c = 1:numel(levels)
    plot(mLog, scaled(:,c), 'LineWidth', 1.6, 'Color', co(c,:));
end
xlabel('log forward moneyness');
ylabel('conditional curve, common-mean scaling');
title(sprintf('%s conditional SDF replication (lambda = %.6g)', ...
    horizon, ms.lambda));
legend(compose('VIX %.0f', levels), 'Location', 'northeast');
hold off;

figFile = fullfile(outDir, sprintf('conditional_sdf_%s.png', horizon));
exportgraphics(f, figFile, 'Resolution', 200);
close(f);

curveTable = table(mLog(:), mRaw, 'VariableNames', {'m_log','m_raw'});
for c = 1:numel(levels)
    curveTable.(sprintf('curve_scaled_v%d', c)) = scaled(:, c);
end
csvFile = fullfile(outDir, sprintf('conditional_sdf_%s.csv', horizon));
writetable(curveTable, csvFile);

outputs = struct('horizon', horizon, 'levels', levels, ...
    'display_labels', {selInfo.display_labels}, 'common_mean', mu, ...
    'figure_file', figFile, 'curve_file', csvFile);

end
