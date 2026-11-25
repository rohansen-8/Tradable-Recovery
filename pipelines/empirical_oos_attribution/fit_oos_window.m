function model_state = fit_oos_window(oos, panel, win)
%% Fit the production estimator on one window's training dates only.
%
% train_projection_model is called as is on the training rows. Three
% configuration fields are redirected for the window, none of which touches
% the estimator mathematics:
%   * the full model state that train_projection_model always saves goes to
%     this run's _scratch folder (overwritten each window), never to
%     results/empirical/;
%   * cfg.real is removed, so build_real_model_meta (which re-reads the full
%     panel) is skipped;
%   * when the maturity's VIX basis range is data driven (weekly), the range
%     is the training rows' min/max instead of the full-panel min/max that
%     config_empirical uses (as in oos_run.py; avoids look-ahead).
% Length scales are option-level medians of the training rows, computed
% inside train_projection_model from the rows passed here.

cfgW = oos.cfg;
if isfield(cfgW, 'real'), cfgW = rmfield(cfgW, 'real'); end
cfgW.dataset.path = oos.paths.scratch;
cfgW.dataset.modelStateFile = fullfile(oos.paths.scratch, ...
    sprintf('%s_window_scratch.mat', oos.horizon));
cfgW.results.path = oos.paths.scratch;
cfgW.results.methodPath = oos.paths.scratch;
cfgW.results.figuresPath = oos.paths.scratch;
cfgW.results.cvCheckpoint = fullfile(oos.paths.scratch, 'unused_cv.mat');
protected = [fullfile(oos.cfg.projectRoot, 'results', 'empirical') filesep];
assert(~startsWith(cfgW.dataset.modelStateFile, protected), ...
    'fit_oos_window:ProtectedPath', 'Scratch save path is inside results/empirical/.');

trainDates = panel.dates(win.train_index);
dfTrain = panel.df(ismember(panel.df.date, trainDates), :);
assert(max(dfTrain.date) < win.oos_date, 'fit_oos_window:LookAhead', ...
    'Training rows reach the OOS date.');

if oos.protocol.exo_range_is_data_driven
    v = dfTrain.VIX;
    cfgW.training.exo_domain = [min(v), max(v)];
    cfgW.training.exo_grid = linspace(min(v), max(v), cfgW.training.nExo).';
end

model_state = train_projection_model(cfgW, dfTrain);
assert(model_state.T == numel(trainDates), 'fit_oos_window:T', ...
    'Fitted T = %d, expected %d training dates.', model_state.T, numel(trainDates));

end
