function r = results_root()
%% Root of all generated results: <project>/results, or SDF_RESULTS_ROOT when set.
% SDF_RESULTS_ROOT redirects every reader and writer at once; run_replication
% uses it to regenerate the empirical outputs into a fresh folder. The reported
% outputs always stay in <project>/results.
r = getenv('SDF_RESULTS_ROOT');
if isempty(r), r = fullfile(get_project_root(), 'results'); end
end
