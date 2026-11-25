function audit = audit_simulation_block_outputs(model, cfg, Tval, Bexpected, raw, reg, recovery, expectRecovery)
%% Check one completed simulation design point before production continues.
%
% The checks are intentionally strict: a failed assertion stops the outer
% production runner after the completed checkpoints have already been saved.

model = upper(string(model));
assert(ismember(model, ["BS","CHJ"]), 'model must be BS or CHJ.');
assert(numel(unique(raw.Tval)) == 1 && unique(raw.Tval) == Tval, ...
    '%s raw output contains the wrong T.', model);

nShapeCells = numel(cfg.testing.regionSpecs) * numel(cfg.testing.testTypes);
assert(height(raw) == Bexpected * nShapeCells, ...
    '%s T=%d raw row count is incorrect.', model, Tval);
assert(numel(unique(raw.seed)) == Bexpected, ...
    '%s T=%d does not contain the requested number of independent seeds.', model, Tval);

[G, ~] = findgroups(raw.seed);
rowsPerSeed = splitapply(@numel, raw.seed, G);
assert(all(rowsPerSeed == nShapeCells), ...
    '%s T=%d has incomplete/duplicated shape rows within a seed.', model, Tval);
assert(all(isfinite(raw.p_value)) && all(raw.p_value >= 0 & raw.p_value <= 1), ...
    '%s T=%d contains an invalid shape-test p-value.', model, Tval);
assert(all(raw.mc_n_valid > 0), ...
    '%s T=%d contains a shape test with no valid Monte Carlo draws.', model, Tval);
assert(all(raw.Tobs == Tval), '%s T=%d contains an incorrect Tobs.', model, Tval);
assert(all(raw.covariance_bandwidth == bartlett_hac_bandwidth(Tval)), ...
    '%s T=%d does not use the frozen Bartlett bandwidth rule.', model, Tval);

lambdaGrid = cfg.training.cv.lambda_grid(:);
if ismember('lambda_fit', raw.Properties.VariableNames)
    lambdaFit = raw.lambda_fit;
else
    lambdaFit = raw.lambda;
end
if ismember('lambda_inference', raw.Properties.VariableNames)
    lambdaInf = raw.lambda_inference;
else
    lambdaInf = raw.lambda;
end
% CV and fixed-lambda rows must lie on the configured grid; eff_dim rows are
% continuous solutions (no grid, no floor) and must be finite and positive.
isEffDim = string(raw.lambda_mode) == "eff_dim";
% fixed_lambda rows must carry exactly the one lambda bound to the design.
isFixedLambda = string(raw.lambda_mode) == "fixed_lambda";
assert(~any(isFixedLambda) || all(isFixedLambda), ...
    '%s T=%d mixes fixed_lambda rows with another lambda rule.', model, Tval);
for j = 1:height(raw)
    if isFixedLambda(j)
        assert(lambdaFit(j) == cfg.training.lambda && lambdaInf(j) == cfg.training.lambda, ...
            '%s T=%d contains a lambda different from the fixed design lambda.', model, Tval);
    elseif isEffDim(j)
        assert(isfinite(lambdaFit(j)) && lambdaFit(j) > 0, ...
            '%s T=%d contains a non-positive eff_dim lambda.', model, Tval);
    else
        assert(min(abs(lambdaGrid - lambdaFit(j))) <= ...
            1e-13 * max(1,abs(lambdaFit(j))), ...
            '%s T=%d contains a fitted lambda outside the configured grid.', model, Tval);
        assert(min(abs(lambdaGrid - lambdaInf(j))) <= ...
            1e-13 * max(1,abs(lambdaInf(j))), ...
            '%s T=%d contains an inference lambda outside the configured grid.', model, Tval);
    end
    lambdaTol = 10 * eps(max(1,max(abs([lambdaFit(j),lambdaInf(j)]))));
    assert(abs(lambdaInf(j)-lambdaFit(j)) <= lambdaTol, '%s T=%d must use lambda_inference = lambda_fit.', model, Tval);
end
if ismember('inference_rule', raw.Properties.VariableNames)
    idxCV = ismember(string(raw.lambda_mode), ["cv","eff_dim","fixed_lambda"]);
    assert(all(string(raw.inference_rule(idxCV)) == string(cfg.testing.inference_lambda.rule)), ...
        '%s T=%d does not use the frozen lambda_inference = lambda_fit rule.', model, Tval);
end
if any(isEffDim)
    assert(ismember('effdim_residual', raw.Properties.VariableNames), ...
        '%s T=%d eff_dim rows are missing their selection diagnostics.', model, Tval);
    E = raw(isEffDim, :);
    assert(all(abs(E.effdim_target - Tval^(1/3) * cfg.training.effdim_c) ...
        <= 1e-12 * max(1, E.effdim_target)), ...
        '%s T=%d eff_dim target differs from effdim_c*T^(1/3).', model, Tval);
    assert(all(abs(E.effdim_residual) < 1e-8 * max(1, E.effdim_target)), ...
        '%s T=%d eff_dim solution misses its target.', model, Tval);
    assert(all(E.effdim_numerical_rank > E.effdim_target), ...
        '%s T=%d eff_dim target is not below the numerical rank.', model, Tval);
end
if ismember('lambda_scale', raw.Properties.VariableNames)
    idxCV = string(raw.lambda_mode) == "cv";
    err = abs(lambdaFit(idxCV) - raw.lambda_multiplier(idxCV).*raw.lambda_scale(idxCV));
    scale = max(1, abs(lambdaFit(idxCV)));
    assert(all(err <= 1e-12 .* scale), ...
        '%s T=%d has inconsistent fitted-lambda metadata.', model, Tval);
end

% Region metadata must match the configured region bounds exactly.
for j = 1:numel(cfg.testing.regionSpecs)
    spec = cfg.testing.regionSpecs(j);
    idx = string(raw.region) == string(spec.name);
    assert(any(idx), '%s T=%d is missing region %s.', model, Tval, spec.name);
    assert(all(abs(raw.region_left(idx) - spec.bounds(1)) < 1e-14) ...
        && all(abs(raw.region_right(idx) - spec.bounds(2)) < 1e-14), ...
        '%s T=%d has stale region bounds for %s.', model, Tval, spec.name);
end

% Production Monte Carlo runs skip per-replication regularization sensitivity.
lambdaPath = cfg.diagnostics.regularization_lambda_path(:);
if cfg.diagnostics.run_per_rep_regularization_path
    assert(height(reg) == Bexpected * nShapeCells * numel(lambdaPath), ...
        '%s T=%d regularization-sensitivity row count is incorrect.', model, Tval);
    assert(numel(unique(reg.seed)) == Bexpected, ...
        '%s T=%d regularization sensitivity has the wrong seed count.', model, Tval);
    assert(numel(unique(reg.lambda)) == numel(lambdaPath) ...
        && all(abs(sort(unique(reg.lambda)) - sort(lambdaPath)) < 1e-14), ...
        '%s T=%d regularization path does not match the frozen lambda path.', model, Tval);
    assert(all(string(reg.distance_metric) == "fixed_cv_hac"), ...
        '%s T=%d regularization sensitivity changed the distance metric.', model, Tval);
else
    assert(isempty(reg), ...
        '%s T=%d unexpectedly ran the disabled per-rep regularization path.', model, Tval);
end

mExpected = build_global_shape_grid(cfg);
if expectRecovery
    assert(height(recovery) == Bexpected * numel(mExpected), ...
        '%s T=%d recovery row count is incorrect.', model, Tval);
    assert(numel(unique(recovery.seed)) == Bexpected, ...
        '%s T=%d recovery has the wrong seed count.', model, Tval);
    mFound = sort(unique(recovery.m));
    assert(numel(mFound) == numel(mExpected) ...
        && max(abs(mFound(:) - mExpected(:))) < 1e-13, ...
        '%s T=%d recovery does not use the exact shape-test grid.', model, Tval);

    [Gr, ~] = findgroups(recovery.seed);
    recoveryRowsPerSeed = splitapply(@numel, recovery.seed, Gr);
    assert(all(recoveryRowsPerSeed == numel(mExpected)), ...
        '%s T=%d recovery is incomplete within a seed.', model, Tval);
    diffMean = splitapply(@mean, recovery.difference, Gr);
    estMean = splitapply(@mean, recovery.estimated_sdf, Gr);
    trueMean = splitapply(@mean, recovery.true_sdf, Gr);
    % Fixed-lambda runs may record undefined (NaN) HJ normalizations; check the rest.
    ok = true(size(estMean));
    if strcmp(cfg.training.lambda_method, 'fixed_lambda'), ok = isfinite(estMean); end
    assert(all(abs(diffMean(ok)) < 1e-9) && all(abs(estMean(ok) - 1) < 1e-9) ...
        && all(abs(trueMean(ok) - 1) < 1e-9), ...
        '%s T=%d recovery normalization check failed.', model, Tval);
else
    assert(isempty(recovery), ...
        '%s T=%d produced recovery rows outside the frozen T=2000 exercise.', model, Tval);
end

lambdaScale = max(ones(size(lambdaFit)), max(abs(lambdaFit),abs(lambdaInf)));
lambdaChanged = abs(lambdaInf-lambdaFit) > 10 .* eps(lambdaScale);

audit = table(model, Tval, Bexpected, height(raw), height(reg), height(recovery), ...
    median(lambdaFit), median(lambdaInf), mean(lambdaChanged), ...
    mean(raw.reject_5pct), min(raw.p_value), max(raw.p_value), ...
    'VariableNames', {'model','Tval','B','shape_rows','regularization_rows', ...
    'recovery_rows','lambda_fit_median','lambda_inference_median', ...
    'lambda_changed_freq','mean_reject_5pct','min_p_value','max_p_value'});

fprintf('\nAUDIT PASSED: %s T=%d, B=%d\n', model, Tval, Bexpected);
disp(audit);

end
