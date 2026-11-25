function test_score_covariance_hac()
%% Verify lag-zero and Bartlett HAC score-covariance construction.
%
% This test confirms the lag-zero covariance convention and that the
% Bartlett bandwidth-sensitivity grid produces finite, symmetric
% covariance matrices of the expected dimension. The L = 0 Bartlett case
% must reproduce the lag-zero S' S / T covariance exactly.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(projectRoot, 'drivers', 'setpath.m'));

rng(11);
S = randn(40, 5);
dates = (1:40).';
T = size(S, 1);
r = size(S, 2);

iidOmega = (S.' * S) / T;
iidOmega = 0.5 * (iidOmega + iidOmega.');

opts = struct();
opts.type = 'iid_lag0';
[Omega0, S0, info0] = build_score_covariance(S, dates, opts);
assert_close(Omega0, iidOmega, 1e-12, 'iid_lag0 must match the iid lag-0 sample covariance.');
assert_close(S0, S, 0, 'iid_lag0 must not alter scores.');
assert(strcmp(info0.type, 'iid_lag0'));
assert(info0.bandwidth == 0);

bandwidth_grid = [0 2 4 6 12];

for ib = 1:numel(bandwidth_grid)
    L = bandwidth_grid(ib);

    opts = struct();
    opts.type = 'hac_bartlett';
    opts.bandwidth = L;
    opts.winsor_margin = 0;
    opts.recenter = false;
    opts.psd_cleanup = L > 0;

    [OmegaH, SH, infoH] = build_score_covariance(S, dates, opts);

    assert(all(size(OmegaH) == [r r]), 'HAC covariance has wrong dimension.');
    assert(all(size(SH) == [T r]), 'Returned score matrix has wrong dimension.');
    assert(all(isfinite(OmegaH(:))), 'HAC covariance contains nonfinite values.');
    assert_close(OmegaH, OmegaH.', 1e-12, 'HAC covariance must be symmetric.');
    assert(strcmp(infoH.type, 'hac_bartlett'));
    assert(infoH.bandwidth == L);
    assert(infoH.requested_bandwidth == L);

    if L == 0
        assert_close(OmegaH, iidOmega, 1e-12, ...
            'HAC bandwidth 0 must match the iid lag-0 sample covariance.');
    end
end

opts = struct();
opts.type = 'hac_bartlett';
opts.bandwidth = 6;
opts.winsor_margin = 0.10;
opts.recenter = true;
opts.psd_cleanup = true;
[OmegaW, SW, infoW] = build_score_covariance(S, dates, opts);

assert(all(isfinite(OmegaW(:))), 'Winsorized HAC covariance contains nonfinite values.');
assert_close(OmegaW, OmegaW.', 1e-12, 'Winsorized HAC covariance must be symmetric.');
assert(all(size(SW) == [T r]), 'Winsorized score matrix has wrong dimension.');
assert(infoW.recentered);
assert(infoW.bandwidth == 6);

fprintf('test_score_covariance_hac passed.\n');
end

function assert_close(A, B, tol, msg)
%% Assert exact or tolerance-based matrix equality.

if tol == 0
    ok = isequal(A, B);
else
    ok = max(abs(A(:) - B(:))) <= tol;
end

if ~ok
    error('test_score_covariance_hac:AssertClose', '%s', msg);
end
end
