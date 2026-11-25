function test_shape_loadings()
%% Verify test identifiers and sign conventions: mono = h_m, conv = -h_mm.

repo = fileparts(fileparts(mfilename('fullpath')));
thetaFile = fullfile(repo, 'testing', 'build_theta_hat.m');
txt = fileread(thetaFile);

assert(contains(txt, "case 'mono'") || contains(txt, 'case ''mono'''), ...
    'build_theta_hat.m must contain case ''mono''.');

assert(contains(txt, "case 'conv'") || contains(txt, 'case ''conv'''), ...
    'build_theta_hat.m must contain case ''conv''.');

badTokens = {'mono_inc','mono_dec','logconv','log_conv'};
for j = 1:numel(badTokens)
    assert(~contains(txt, badTokens{j}), ...
        'Forbidden test identifier found in build_theta_hat.m: %s', badTokens{j});
end

assert(contains(txt, 'd = -[k12; k22]') || contains(txt, 'd=-[k12;k22]'), ...
    'conv loading must use -h_mm, implemented as d = -[k12; k22].');

assert(~contains(txt, 'm *') && ~contains(txt, 'm*'), ...
    'build_theta_hat.m appears to contain a log-moneyness chain-rule term.');

fprintf('test_shape_loadings passed.\n');
end
