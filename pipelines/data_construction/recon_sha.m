function h = recon_sha(f)
%% SHA-256 of a file via the system shasum tool.
[s, o] = system(sprintf('shasum -a 256 "%s"', f));
assert(s == 0, 'shasum failed on %s', f);
h = strtok(o);
end
