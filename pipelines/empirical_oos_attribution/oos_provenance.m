function prov = oos_provenance(oos)
%% Record code and data provenance for an OOS run.
%
% Every MATLAB file the OOS computation depends on (found by
% requiredFilesAndProducts from the pipeline entry points) is hashed with
% SHA-256 and its Git state recorded. code_hash summarizes those hashes and
% enters every window fingerprint, so a window computed under different code
% is never silently reused. is_clean is true only when every dependency is
% tracked and unmodified in its repository: the production gate in
% run_empirical_oos_attribution requires it.

entry = {'run_empirical_oos_attribution','fit_oos_window', ...
    'evaluate_oos_payoff_components','load_oos_panel','build_oos_schedule', ...
    'config_empirical_oos','oos_sha256'};
files = matlab.codetools.requiredFilesAndProducts(entry);
files = sort(files(endsWith(files, '.m')));

n = numel(files);
rec = struct('file', cell(n,1), 'sha256', [], 'git_status', [], 'repo', []);
for j = 1:n
    rec(j).file = files{j};
    rec(j).sha256 = sha256_file(files{j});
    [rec(j).repo, rec(j).git_status] = git_file_status(files{j});
end
keys = arrayfun(@(r) sprintf('%s:%s', r.file, r.sha256), rec, 'UniformOutput', false);

prov = struct();
prov.created_on = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
prov.matlab_version = version;
prov.files = rec;
prov.code_hash = oos_sha256(strjoin(keys, newline));
prov.is_clean = all(arrayfun(@(r) isempty(r.git_status), rec));
prov.dirty_files = {rec(~arrayfun(@(r) isempty(r.git_status), rec)).file};
prov.data_file = oos.cfg.dataset.dfFile;
prov.data_sha256 = sha256_file(oos.cfg.dataset.dfFile);
prov.repos = unique({rec.repo});
prov.repo_heads = cellfun(@git_head, prov.repos, 'UniformOutput', false);

end

function h = sha256_file(f)
%% SHA-256 of a file's bytes.
fid = fopen(f, 'r');
assert(fid > 0, 'oos_provenance:Open', 'Cannot open %s.', f);
c = onCleanup(@() fclose(fid));
h = oos_sha256(fread(fid, Inf, '*uint8'));
end

function [repo, status] = git_file_status(f)
%% Repository top level and porcelain status ('' = tracked and clean).
d = fileparts(f);
[s1, repo] = system(sprintf('git -C "%s" rev-parse --show-toplevel', d));
assert(s1 == 0, 'oos_provenance:Git', 'Not in a Git repository: %s.', f);
repo = strtrim(repo);
[~, status] = system(sprintf('git -C "%s" status --porcelain -- "%s"', d, f));
status = strtrim(status);
end

function s = git_head(repo)
%% Branch and commit of a repository.
[~, b] = system(sprintf('git -C "%s" rev-parse --abbrev-ref HEAD', repo));
[~, c] = system(sprintf('git -C "%s" rev-parse HEAD', repo));
s = sprintf('%s @ %s', strtrim(b), strtrim(c));
end
