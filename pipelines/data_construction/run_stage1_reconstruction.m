function run_stage1_reconstruction()
%% Option-panel reconstruction driver: extract options once, build baseline and forward-diagnostic panels, compare, diagnose.
%
% Restartable: each step is skipped if its output exists (outputs are never
% overwritten). Expensive pass: one chunked read of the options CSV.
c = recon_config();
cand = fullfile(c.extracted, 'options_candidates.mat');
if ~isfile(cand), recon_extract_options(); end
if ~isfolder(fullfile(c.baseline, 'daily')), recon_build_panels('full', 'none'); end
fwd = fullfile(c.diagnostics, 'forward_anomaly', 'exclude_scale_errors');
if ~isfolder(fullfile(fwd, 'daily')), recon_build_panels('full', 'exclude_scale_errors', fwd); end
tables = fullfile(c.reports, 'tables');
rows = fullfile(c.diagnostics, 'panel_comparison');
if ~isfile(fullfile(tables, 'stage1_compare_counts.csv'))
    recon_compare_panels(c.baseline, tables, rows, cand);
end
recon_diagnostics(c.baseline, fwd, rows, c.diagnostics, tables);
end
