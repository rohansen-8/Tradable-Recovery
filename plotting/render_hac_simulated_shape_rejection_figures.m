function outputs = render_hac_simulated_shape_rejection_figures(saveFigure, lambdaMethod)
%% Render BS and CHJ HAC simulation rejection-frequency figures.
%
% Reads the saved simulation summaries only (no simulation, fitting or
% inference) and renders Figures A.3--A.4 style rejection-probability panels
% (local function plot_size_study_rejection_panel). lambdaMethod ('eff_dim' default,
% get_frozen_design().simulation_lambda_method, or 'cv') selects which saved
% results are read and where the figures are written.

if nargin < 1 || isempty(saveFigure)
    saveFigure = true;
end
if nargin < 2 || isempty(lambdaMethod)
    lambdaMethod = get_frozen_design().simulation_lambda_method;
end

%% Load final simulation summaries
simRoot = simulation_output_dir('simulations', lambdaMethod);
bsFile = fullfile(simRoot, 'bs_increasing_t', 'bs_results.mat');
chjFile = fullfile(simRoot, 'chj_increasing_t', 'chj_results.mat');

BS = load(bsFile, 'rejection_summary');
CHJ = load(chjFile, 'rejection_summary');

bsTbl = BS.rejection_summary( ...
    strcmpi(string(BS.rejection_summary.design_group), 'monthly_performance'), :);

chjTbl = CHJ.rejection_summary( ...
    strcmpi(string(CHJ.rejection_summary.design_group), 'daily_one_day_T_sequence'), :);

if isempty(bsTbl)
    error('render_hac_simulated_shape_rejection_figures:MissingBS', ...
        'No BS monthly_performance rows found.');
end

if isempty(chjTbl)
    error('render_hac_simulated_shape_rejection_figures:MissingCHJ', ...
        'No CHJ daily_one_day_T_sequence rows found.');
end

%% Render panels
% Manuscript layout (1%/5%/10% legend, wide panel, small tick labels) with a
% data-adaptive y-axis: upper limit = max rejection rounded up to 0.05, so
% rates near the nominal levels stay readable.
panelOpts = {'Alpha', 'all', 'ShowSE', false, 'ShowTitle', false, 'Visible', 'off', ...
    'YMax', [], 'ShowLegend', true, 'Layout', 'manuscript'};
figBS = plot_size_study_rejection_panel(bsTbl, 'Black-Scholes', panelOpts{:});
figCHJ = plot_size_study_rejection_panel(chjTbl, 'CHJ', panelOpts{:});

%% Save figure files
outDir = fullfile(simulation_output_dir('figures', lambdaMethod), 'simulation_size_studies');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

bsPdf = fullfile(outDir, 'figure_A3_BS_hac_rejection_probabilities.pdf');
bsPng = fullfile(outDir, 'figure_A3_BS_hac_rejection_probabilities.png');
chjPdf = fullfile(outDir, 'figure_A4_CHJ_hac_rejection_probabilities.pdf');
chjPng = fullfile(outDir, 'figure_A4_CHJ_hac_rejection_probabilities.png');

if saveFigure
    exportgraphics(figBS, bsPdf, 'ContentType', 'vector');
    exportgraphics(figBS, bsPng, 'Resolution', 300);
    exportgraphics(figCHJ, chjPdf, 'ContentType', 'vector');
    exportgraphics(figCHJ, chjPng, 'Resolution', 300);

    fprintf('\nSaved BS figure:\n  %s\n  %s\n', bsPdf, bsPng);
    fprintf('Saved CHJ figure:\n  %s\n  %s\n', chjPdf, chjPng);
end

outputs = table( ...
    ["BS"; "CHJ"], ...
    [string(bsPdf); string(chjPdf)], ...
    [string(bsPng); string(chjPng)], ...
    'VariableNames', {'design', 'pdfFile', 'pngFile'});

end

function fig = plot_size_study_rejection_panel(summaryTbl, modelLabel, varargin)
%% Plot simulation rejection frequencies in a 3-by-3 panel.
%
% Paper-style plot matching manuscript Figures A.3--A.4
% (pics/montecarlo/{BS,CHJ}_shape_tests.png):
% rows are monotonicity/convexity/concavity;
% columns are OTM put/near ATM/OTM call;
% Alpha='all' plots 1%, 5%, and 10% rejection probabilities.
%
% Options:
%   'YMax'       : fixed upper y-limit (e.g. 0.7, the manuscript axis). With
%                  0.7 the ticks are 0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7. Data
%                  above YMax are an error, never clipped. Default [] keeps
%                  the data-adaptive axis.
%   'ShowLegend' : 1%/5%/10% legend in the top-left panel (Alpha='all' only).
%   'Layout'     : 'manuscript' (8.9 x 5.65 in, small tick labels, legend
%                  style of the published figures) or 'compact'
%                  (7.2 x 5.2 in). Default 'compact'.


%% Parse options
p = inputParser;
addRequired(p, 'summaryTbl', @istable);
addRequired(p, 'modelLabel', @(x) ischar(x) || isstring(x));
addParameter(p, 'Alpha', 'all', @(x) ischar(x) || isstring(x));
addParameter(p, 'ShowSE', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'ShowTitle', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'Visible', 'on', @(x) ischar(x) || isstring(x));
addParameter(p, 'YMax', [], @(x) isempty(x) || (isscalar(x) && x > 0));
addParameter(p, 'ShowLegend', false, @(x) islogical(x) && isscalar(x));
addParameter(p, 'Layout', 'compact', @(x) any(strcmpi(string(x), ["compact", "manuscript"])));
parse(p, summaryTbl, modelLabel, varargin{:});

alphaTag = lower(string(p.Results.Alpha));
showSE = p.Results.ShowSE;
showTitle = p.Results.ShowTitle;
figVisible = char(p.Results.Visible);
plotAll = alphaTag == "all";
showLegend = p.Results.ShowLegend && plotAll;
isManuscript = strcmpi(string(p.Results.Layout), "manuscript");

summaryTbl = normalize_size_study_table(summaryTbl);

maxRej = max([summaryTbl.rej_1pct; summaryTbl.rej_5pct; summaryTbl.rej_10pct]);
if isempty(p.Results.YMax)
    % Keep the data below ~75% of the axis height so the in-panel restriction
    % labels and the legend stay clear of the curves; an all-zero panel
    % still gets a valid axis.
    yMax = max(ceil(20 * maxRej / 0.75) / 20, 0.05);
    if yMax <= 0.25
        yTicks = 0:0.05:yMax;
    else
        yTicks = [0 0.01 0.05 0.10 0.30 0.50 0.70 0.90];
    end
else
    yMax = p.Results.YMax;
    assert(maxRej <= yMax, 'plot_size_study_rejection_panel:AboveYMax', ...
        'A rejection frequency (%.4f) exceeds the fixed YMax %.4f.', maxRej, yMax);
    yTicks = [0 0.01 0.05 0.10 0.30 0.50 0.70 0.90];
    yTicks = yTicks(yTicks <= yMax + 1e-12);
end

if isManuscript
    figSize = [8.9 5.65];
    fsTitle = 15; fsPanel = 15; fsLabel = 15; fsTick = 10; fsLegend = 15;
    lineW = 1.2; markerS = 3.5; axisW = 0.75;
    refColor = [0.45 0.45 0.45]; refW = 0.6;
else
    figSize = [7.2 5.2];
    fsTitle = 16; fsPanel = 15; fsLabel = 14; fsTick = 13; fsLegend = 13;
    lineW = 1.5; markerS = 4; axisW = 1;
    refColor = [0 0 0]; refW = 0.55;
end




%% Panel layout
Tvals = [500 1000 2000 5000 10000];
x = 1:numel(Tvals);
xLabs = ["500","1000","2000","5000","10000"];

testKeys = ["mono", "conv", "conc"];
testLabs = ["Monotonicity", "Convexity", "Concavity"];
regionKeys = ["left", "center", "right"];
regionLabs = ["OTM Put", "Near ATM", "OTM Call"];

if plotAll
    alphaTags = ["1pct", "5pct", "10pct"];
else
    alphaTags = alphaTag;
end

lineColors = [0.0000 0.4470 0.7410;
              0.6350 0.0780 0.1840;
              45/255 80/255 22/255];
markers = ["o", "s", "^"];


%% Draw rejection-probability panel
fig = figure('Color', 'w', 'Visible', figVisible, 'Units', 'inches', ...
    'Position', [1 1 figSize], 'PaperUnits', 'inches', 'PaperSize', figSize);
legendHandles = gobjects(0);
tlo = tiledlayout(fig, 3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for iTest = 1:numel(testKeys)
    for iRegion = 1:numel(regionKeys)
        ax = nexttile(tlo);
        hold(ax, 'on');

        idx = summaryTbl.test_plot == testKeys(iTest) & summaryTbl.region_plot == regionKeys(iRegion);
        cellTbl = summaryTbl(idx, :);

        for a = [0.01 0.05 0.10]
            yline(ax, a, '--', 'Color', refColor, 'LineWidth', refW, ...
                'HandleVisibility', 'off');
        end

        for iAlpha = 1:numel(alphaTags)
            [rejVar, seVar] = rate_columns(summaryTbl, alphaTags(iAlpha));
            y = nan(size(Tvals));
            e = nan(size(Tvals));

            for iT = 1:numel(Tvals)
                row = cellTbl(cellTbl.T_plot == Tvals(iT), :);
                if ~isempty(row)
                    y(iT) = row.(rejVar)(1);
                    if showSE && ~isempty(seVar)
                        e(iT) = row.(seVar)(1);
                    end
                end
            end

            kColor = iAlpha;
            if ~plotAll
                kColor = alpha_color_index(alphaTags(iAlpha));
            end

            lineSpec = ['-' char(markers(kColor))];
            if showSE && ~isempty(seVar) && ~plotAll
                errorbar(ax, x, y, e, lineSpec, ...
                    'Color', lineColors(kColor,:), ...
                    'MarkerFaceColor', 'w', ...
                    'MarkerSize', markerS, ...
                    'LineWidth', lineW, ...
                    'CapSize', 3);
            else
                hLine = plot(ax, x, y, lineSpec, ...
                    'Color', lineColors(kColor,:), ...
                    'MarkerFaceColor', 'w', ...
                    'MarkerSize', markerS, ...
                    'LineWidth', lineW);
                if iTest == 1 && iRegion == 1
                    legendHandles(end+1) = hLine; %#ok<AGROW>
                end
            end
        end

        if iTest == 1
            title(ax, regionLabs(iRegion), ...
                'FontName', 'Times New Roman', ...
                'FontSize', fsTitle, ...
                'FontWeight', 'normal');
        end

        text(ax, 0.07, 0.84, testLabs(iTest), ...
            'Units', 'normalized', ...
            'FontName', 'Times New Roman', ...
            'FontSize', fsPanel, 'FontWeight', 'normal');

        if iRegion == 1
            ylabel(ax, 'Rejection prob.', ...
                'FontName', 'Times New Roman', ...
                'FontSize', fsLabel);
        end
        if iTest == numel(testKeys)
            xlabel(ax, 'Sample size T', ...
                'FontName', 'Times New Roman', ...
                'FontSize', fsLabel);
        end

        set(ax, ...
            'FontName', 'Times New Roman', ...
            'FontSize', fsTick, ...
            'LineWidth', axisW, ...
            'TickDir', 'in', ...
            'Box', 'on', ...
            'XLim', [0.65 numel(Tvals)+0.35], ...
            'XTick', x, ...
            'XTickLabel', xLabs, ...
            'XTickLabelRotation', 0, ...
            'YLim', [0 yMax], ...
            'YTick', yTicks);
        grid(ax, 'off');
        if isManuscript
            % FontSize scales the axis labels and titles with the tick labels;
            % restore the larger label/title sizes after setting the ticks.
            ax.Title.FontSize = fsTitle;
            ax.XLabel.FontSize = fsLabel;
            ax.YLabel.FontSize = fsLabel;
        end
        if showLegend && iTest == 1 && iRegion == 1 && ~isempty(legendHandles)
            legend(ax, legendHandles, ["1%", "5%", "10%"], ...
                'Location', 'northeast', 'Box', 'off', ...
                'FontName', 'Times New Roman', 'FontSize', fsLegend, ...
                'Interpreter', 'none', 'AutoUpdate', 'off');
        end
        hold(ax, 'off');
    end
end

if showTitle
    sgtitle(tlo, sprintf('%s empirical rejection probabilities', string(modelLabel)), ...
        'FontName', 'Times New Roman', ...
        'FontSize', 9.5, ...
        'FontWeight', 'normal');
end

end


%% Local helpers
function tbl = normalize_size_study_table(tbl)
vars = string(tbl.Properties.VariableNames);

tVar = first_existing_var(vars, ["T", "Tval", "Tvals", "Tobs", "sample_size", "sampleSize", "nObs"]);
testVar = first_existing_var(vars, ["testType", "test_type", "test", "shape", "restriction"]);
regionVar = first_existing_var(vars, ["region", "regionName", "moneynessRegion"]);

tbl.T_plot = double(tbl.(tVar));
tbl.test_plot = clean_string_column(tbl.(testVar));
tbl.region_plot = clean_string_column(tbl.(regionVar));

tbl.region_plot(tbl.region_plot == "otm put") = "left";
tbl.region_plot(tbl.region_plot == "near atm") = "center";
tbl.region_plot(tbl.region_plot == "otm call") = "right";

tbl.test_plot(tbl.test_plot == "monotonicity") = "mono";
tbl.test_plot(tbl.test_plot == "convexity") = "conv";
tbl.test_plot(tbl.test_plot == "concavity") = "conc";

end

function v = first_existing_var(vars, candidates)
varsLower = lower(vars);
for i = 1:numel(candidates)
    idx = find(varsLower == lower(candidates(i)), 1);
    if ~isempty(idx)
        v = char(vars(idx));
        return;
    end
end
error('Required column not found. Available columns: %s', strjoin(vars, ', '));
end

function [rejVar, seVar] = rate_columns(tbl, alphaTag)
vars = string(tbl.Properties.VariableNames);
rejVar = "rej_" + string(alphaTag);
seVar = "se_" + string(alphaTag);

if ~ismember(rejVar, vars)
    error('plot_size_study_rejection_panel:MissingColumn', ...
        'Missing required column %s.', rejVar);
end
if ~ismember(seVar, vars)
    seVar = "";
end

rejVar = char(rejVar);
seVar = char(seVar);
end

function k = alpha_color_index(alphaTag)
if alphaTag == "1pct"
    k = 1;
elseif alphaTag == "5pct"
    k = 2;
elseif alphaTag == "10pct"
    k = 3;
else
    k = 1;
end
end

function s = clean_string_column(x)
s = string(x);
s = erase(s, "{");
s = erase(s, "}");
s = erase(s, char(39));
s = strip(lower(s));
end
