function meta = simulation_run_metadata(lambdaMethod)
%% Provenance record saved with every simulation checkpoint and result file.
%
% Records the MATLAB release, the Git commit of the code that produced the
% file, whether the working tree had uncommitted changes (and a hash of the
% tracked-file diff, so a dirty run is still identifiable), the lambda rule
% and a UTC timestamp. Metadata only: nothing here enters a computation or a
% design fingerprint, so it never affects checkpoint resumption.

if nargin < 1 || isempty(lambdaMethod), lambdaMethod = ''; end
root = get_project_root();
git = @(args) local_git(root, args);

meta = struct();
meta.matlab_version = string(version);
meta.matlab_release = string(version('-release'));
meta.git_commit = git('rev-parse HEAD');
meta.git_branch = git('rev-parse --abbrev-ref HEAD');
status = git('status --porcelain --untracked-files=no');
meta.git_dirty = strlength(status) > 0;
diffText = git('diff HEAD');
if strlength(diffText) > 0
    md = java.security.MessageDigest.getInstance('SHA-1');
    md.update(uint8(char(diffText)));
    meta.git_diff_sha1 = string(lower(reshape(dec2hex(typecast(md.digest(), 'uint8')).', 1, [])));
else
    meta.git_diff_sha1 = "";
end
meta.lambda_method = string(lambdaMethod);
meta.created_utc = string(datetime('now', 'TimeZone', 'UTC', ...
    'Format', 'yyyy-MM-dd''T''HH:mm:ss''Z'''));

end

function out = local_git(root, args)
[status, txt] = system(sprintf('git -C "%s" %s', root, args));
if status ~= 0, out = "unavailable"; else, out = strtrim(string(txt)); end
end
