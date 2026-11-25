function cv = cross_validate_lambda(cfg, alphaPanel, K_h)
%% Select lambda by one-step-ahead expanding-window projection loss.
% Search interval [1e-5, 1e-2], searched exhaustively over the fixed grid
% logspace(-5,-2,31).

    alpha = alphaPanel.alpha;
    T = size(alpha, 2);

    multiplier_grid = get_lambda_grid(cfg);
    scaleMode = get_lambda_scale_mode(cfg);
    minTrain = get_min_train_dates(cfg, T);
    valIdx = (minTrain + 1):T;
    solver = get_solver(cfg);

    alpha0 = alpha(:, 1:minTrain);
    alpha_sum0 = sum(alpha0, 2);
    Q_sum0 = alpha0 * alpha0.';
    Q_sum0 = 0.5 * (Q_sum0 + Q_sum0.');

    switch solver
        case 'direct'
            [mean_loss, n_valid_dates, loss_by_date] = direct_grid_loss( ...
                cfg, alpha, K_h, multiplier_grid, minTrain, alpha_sum0, Q_sum0, scaleMode);
        case 'fold_eig'
            [mean_loss, n_valid_dates, loss_by_date] = fold_eig_grid_loss( ...
                alpha, K_h, multiplier_grid, minTrain, alpha_sum0, Q_sum0, scaleMode);
        case 'fold_eig_prewhiten'
            [mean_loss, n_valid_dates, loss_by_date] = fold_eig_prewhiten_grid_loss( ...
                alpha, K_h, multiplier_grid, minTrain, scaleMode);
        case 'fold_eig_woodbury'
            refreshEvery = get_woodbury_refresh(cfg);
            [mean_loss, n_valid_dates, loss_by_date] = fold_eig_woodbury_grid_loss( ...
                alpha, K_h, multiplier_grid, minTrain, refreshEvery, scaleMode);
        otherwise
            error('cross_validate_lambda:BadSolver', ...
                'Unknown CV solver: %s.', solver);
    end

    Q_full = (alpha * alpha.') / T;
    Q_full = 0.5 * (Q_full + Q_full.');
    lambdaScale = regularization_scale(K_h, Q_full, scaleMode);
    lambda_grid = multiplier_grid * lambdaScale;

    % Optional diagnostic-only restriction on the candidates eligible to win CV.
    % Extra lower-lambda points may then be evaluated for inference without
    % changing the ordinary fitted-model selection set.
    selectionMask = true(size(mean_loss));
    if isfield(cfg.training.cv, 'selection_lambda_min') && ...
            ~isempty(cfg.training.cv.selection_lambda_min)
        assert(strcmp(scaleMode, 'absolute'), ...
            'selection_lambda_min currently requires absolute lambda scaling.');
        selectionMin = double(cfg.training.cv.selection_lambda_min);
        assert(isscalar(selectionMin) && isfinite(selectionMin) && selectionMin > 0, ...
            'selection_lambda_min must be a positive finite scalar.');
        selectionMask = selectionMask & lambda_grid >= selectionMin * (1 - 1e-12);
    else
        selectionMin = NaN;
    end
    assert(any(selectionMask), 'No lambda candidates are eligible for CV selection.');
    eligibleIdx = find(selectionMask);
    [~, idxLocal] = min(mean_loss(eligibleIdx));
    idx = eligibleIdx(idxLocal);

    cv = struct();
    cv.lambda = lambda_grid(idx);
    cv.lambda_grid = lambda_grid(:);
    cv.lambda_multiplier = multiplier_grid(idx);
    cv.lambda_multiplier_grid = multiplier_grid(:);
    cv.lambda_scale = lambdaScale;
    cv.lambda_scale_mode = scaleMode;
    cv.selected_index = idx;
    cv.selection_lambda_min = selectionMin;
    cv.selection_mask = selectionMask(:);
    cv.method = '1_step_ahead';
    cv.objective = 'projection_loss';
    cv.selection_rule = 'mean_loss';
    cv.solver = solver;
    cv.min_train_dates = minTrain;
    cv.valIdx = valIdx(:);
    cv.mean_loss = mean_loss;
    cv.n_valid_dates = n_valid_dates;
    cv.loss_by_date = loss_by_date;
    cv.results = table(multiplier_grid(:), lambda_grid(:), mean_loss, n_valid_dates, ...
        'VariableNames', {'lambda_multiplier', 'lambda', 'mean_loss', 'n_valid_dates'});
end

function [mean_loss, n_valid_dates, loss_by_date] = direct_grid_loss( ...
    cfg, alpha, K_h, multiplier_grid, minTrain, alphaSum0, QSum0, scaleMode)
%% Evaluate regularization candidates with direct solves.

    nLambda = numel(multiplier_grid);
    nVal = size(alpha, 2) - minTrain;
    mean_loss = Inf(nLambda, 1);
    n_valid_dates = zeros(nLambda, 1);
    loss_by_date = NaN(nVal, nLambda);
    usePar = use_parallel(cfg) && nLambda > 1;

    if usePar
        parfor j = 1:nLambda
            [mean_loss(j), n_valid_dates(j), loss_j] = cv_loss_for_lambda( ...
                alpha, K_h, multiplier_grid(j), minTrain, alphaSum0, QSum0, scaleMode);
            loss_by_date(:, j) = loss_j;
        end
    else
        for j = 1:nLambda
            [mean_loss(j), n_valid_dates(j), loss_j] = cv_loss_for_lambda( ...
                alpha, K_h, multiplier_grid(j), minTrain, alphaSum0, QSum0, scaleMode);
            loss_by_date(:, j) = loss_j;
        end
    end
end

function [mean_loss, n_valid_dates, loss_by_date] = fold_eig_grid_loss( ...
    alpha, K_h, multiplier_grid, minTrain, alphaSum, QSum, scaleMode)
%% Evaluate all regularization multipliers with one eigendecomposition per fold.

    [L, flag] = chol(0.5 * (K_h + K_h.'), 'lower');
    if flag ~= 0
        error('cross_validate_lambda:FoldEigNeedsSPD', ...
            'fold_eig requires positive-definite K_h.');
    end

    T = size(alpha, 2);
    nLambda = numel(multiplier_grid);
    loss_sum = zeros(nLambda, 1);
    n_valid_dates = zeros(nLambda, 1);
    loss_by_date = NaN(T - minTrain, nLambda);

    for t = (minTrain + 1):T
        nTrain = t - 1;
        alpha_bar = alphaSum / nTrain;
        Q_alpha = QSum / nTrain;
        Q_alpha = 0.5 * (Q_alpha + Q_alpha.');

        B = L \ Q_alpha / L.';
        B = 0.5 * (B + B.');
        [V, D] = eig(B);
        d = real(diag(D));
        V = real(V);

        c = V.' * (L \ alpha_bar);
        x = V.' * (L \ alpha(:, t));
        numer = x .* c;

        lambdaScale = regularization_scale_from_whitened(B, scaleMode);
        lambdaFold = multiplier_grid * lambdaScale;
        denom = d + lambdaFold.';
        prediction = sum(numer ./ denom, 1).';
        loss = (1 - prediction) .^ 2;
        loss_by_date(t - minTrain, :) = loss.';

        good = isfinite(loss);
        loss_sum(good) = loss_sum(good) + loss(good);
        n_valid_dates(good) = n_valid_dates(good) + 1;

        alpha_t = alpha(:, t);
        alphaSum = alphaSum + alpha_t;
        QSum = QSum + alpha_t * alpha_t.';
    end

    minValid = max(2, floor(0.5 * (T - minTrain)));
    mean_loss = Inf(nLambda, 1);
    goodLambda = n_valid_dates >= minValid;
    mean_loss(goodLambda) = loss_sum(goodLambda) ./ n_valid_dates(goodLambda);
end

function [mean_loss, n_valid_dates, loss_by_date] = fold_eig_prewhiten_grid_loss(alpha, K_h, multiplier_grid, minTrain, scaleMode)
%% Evaluate all regularization multipliers after whitening the alpha panel once.

    [L, flag] = chol(0.5 * (K_h + K_h.'), 'lower');
    if flag ~= 0
        error('cross_validate_lambda:FoldEigNeedsSPD', ...
            'fold_eig_prewhiten requires positive-definite K_h.');
    end

    Z = L \ alpha;
    T = size(Z, 2);
    nLambda = numel(multiplier_grid);

    Z0 = Z(:, 1:minTrain);
    zSum = sum(Z0, 2);
    ZZSum = Z0 * Z0.';
    ZZSum = 0.5 * (ZZSum + ZZSum.');

    loss_sum = zeros(nLambda, 1);
    n_valid_dates = zeros(nLambda, 1);
    loss_by_date = NaN(T - minTrain, nLambda);

    for t = (minTrain + 1):T
        nTrain = t - 1;

        b = zSum / nTrain;
        B = ZZSum / nTrain;
        B = 0.5 * (B + B.');

        [V, D] = eig(B);
        d = real(diag(D));
        V = real(V);

        c = V.' * b;
        x = V.' * Z(:, t);
        numer = x .* c;

        lambdaScale = regularization_scale_from_whitened(B, scaleMode);
        lambdaFold = multiplier_grid * lambdaScale;
        denom = d + lambdaFold.';
        prediction = sum(numer ./ denom, 1).';
        loss = (1 - prediction).^2;
        loss_by_date(t - minTrain, :) = loss.';

        good = isfinite(loss);
        loss_sum(good) = loss_sum(good) + loss(good);
        n_valid_dates(good) = n_valid_dates(good) + 1;

        z_t = Z(:, t);
        zSum = zSum + z_t;
        ZZSum = ZZSum + z_t * z_t.';
    end

    minValid = max(2, floor(0.5 * (T - minTrain)));
    mean_loss = Inf(nLambda, 1);
    goodLambda = n_valid_dates >= minValid;
    mean_loss(goodLambda) = loss_sum(goodLambda) ./ n_valid_dates(goodLambda);
end

function [mean_loss, n_valid_dates, loss_by_date] = fold_eig_woodbury_grid_loss( ...
    alpha, K_h, multiplier_grid, minTrain, refreshEvery, scaleMode)
%% Evaluate the same scaled CV systems with periodic EVD refreshes and Woodbury updates.

    [L, flag] = chol(0.5 * (K_h + K_h.'), 'lower');
    if flag ~= 0
        error('cross_validate_lambda:FoldEigNeedsSPD', ...
            'fold_eig_woodbury requires positive-definite K_h.');
    end

    Z = L \ alpha;
    [p, T] = size(Z);
    nLambda = numel(multiplier_grid);

    Z0 = Z(:, 1:minTrain);
    zSum = sum(Z0, 2);
    ZZSum = Z0 * Z0.';
    ZZSum = 0.5 * (ZZSum + ZZSum.');

    loss_sum = zeros(nLambda, 1);
    n_valid_dates = zeros(nLambda, 1);
    loss_by_date = NaN(T - minTrain, nLambda);

    V = [];
    d = [];
    g0 = [];
    W = zeros(p, 0);

    for t = (minTrain + 1):T
        nTrain = t - 1;

        if isempty(V) || size(W, 2) >= refreshEvery
            S0 = 0.5 * (ZZSum + ZZSum.');
            [V, D] = eig(S0);
            d = real(diag(D));
            V = real(V);
            g0 = V.' * zSum;
            W = zeros(p, 0);
        end

        q = V.' * Z(:, t);
        g = g0;
        if ~isempty(W)
            g = g + sum(W, 2);
        end

        prediction = NaN(nLambda, 1);
        Btrace = trace(ZZSum) / nTrain;
        lambdaScale = regularization_scale_from_trace(Btrace, p, scaleMode);
        lambdaFold = multiplier_grid * lambdaScale;

        for j = 1:nLambda
            denom = d + nTrain * lambdaFold(j);

            if any(~isfinite(denom)) || any(denom <= 0)
                continue;
            end

            invDenom = 1 ./ denom;
            a = invDenom .* g;
            pred = q.' * a;

            if ~isempty(W)
                DW = invDenom .* W;
                M = eye(size(W, 2)) + W.' * DW;
                M = 0.5 * (M + M.');

                rhs = W.' * a;
                [R, cholFlag] = chol(M, 'lower');
                assert(cholFlag == 0, ...
                    'cross_validate_lambda:WoodburyNotSPD', ...
                    'Woodbury small matrix lost positive definiteness.');
                coeff = R.' \ (R \ rhs);

                pred = pred - (q.' * DW) * coeff;
            end

            prediction(j) = real(pred);
        end

        loss = (1 - prediction).^2;
        loss_by_date(t - minTrain, :) = loss.';
        good = isfinite(loss);
        loss_sum(good) = loss_sum(good) + loss(good);
        n_valid_dates(good) = n_valid_dates(good) + 1;

        z_t = Z(:, t);
        zSum = zSum + z_t;
        ZZSum = ZZSum + z_t * z_t.';
        W(:, end + 1) = q;
    end

    minValid = max(2, floor(0.5 * (T - minTrain)));
    mean_loss = Inf(nLambda, 1);
    goodLambda = n_valid_dates >= minValid;
    mean_loss(goodLambda) = loss_sum(goodLambda) ./ n_valid_dates(goodLambda);
end

function [meanLoss, nValidDates, loss_t] = cv_loss_for_lambda( ...
    alpha, K_h, multiplier, minTrain, alphaSum, QSum, scaleMode)
%% Evaluate one regularization multiplier on the expanding one-step sequence.

    T = size(alpha, 2);
    loss_t = NaN(T - minTrain, 1);

    for t = (minTrain + 1):T
        nTrain = t - 1;

        alpha_bar = alphaSum / nTrain;
        Q_alpha = QSum / nTrain;
        Q_alpha = 0.5 * (Q_alpha + Q_alpha.');

        lambdaScale = regularization_scale(K_h, Q_alpha, scaleMode);
        lambda = multiplier * lambdaScale;
        A = Q_alpha + lambda * K_h;
        A = 0.5 * (A + A.');

        h_star = solve_system(A, alpha_bar);
        replicating_portfolio = alpha(:, t).' * h_star;
        loss_t(t - minTrain) = (1 - replicating_portfolio)^2;

        alpha_t = alpha(:, t);
        alphaSum = alphaSum + alpha_t;
        QSum = QSum + alpha_t * alpha_t.';
    end

    goodLoss = loss_t(isfinite(loss_t));
    nValidDates = numel(goodLoss);

    if nValidDates < max(2, floor(0.5 * numel(loss_t)))
        meanLoss = Inf;
        return;
    end

    meanLoss = mean(goodLoss);
end

function h_star = solve_system(A, b)
%% Solve the regularized least-squares system.

    [L, flag] = chol(A, 'lower');
    if flag == 0
        h_star = L' \ (L \ b);
    else
        h_star = A \ b;
    end
end

function lambda_grid = get_lambda_grid(cfg)
%% Read the deterministic MATLAB lambda search grid.

    if isfield(cfg.training, 'cv') && isfield(cfg.training.cv, 'lambda_grid') ...
            && ~isempty(cfg.training.cv.lambda_grid)
        lambda_grid = cfg.training.cv.lambda_grid(:);
    else
        lambda_grid = logspace(-5, -2, 31).';
    end

    assert(all(isfinite(lambda_grid)) && all(lambda_grid > 0), ...
        'Lambda candidates must be finite and strictly positive.');
end

function mode = get_lambda_scale_mode(cfg)
%% Read how the dimensionless CV grid is mapped to the penalty scale.

    mode = 'absolute';
    if isfield(cfg.training, 'cv') && isfield(cfg.training.cv, 'lambda_scale_mode') ...
            && ~isempty(cfg.training.cv.lambda_scale_mode)
        mode = char(lower(string(cfg.training.cv.lambda_scale_mode)));
    end

    valid = {'absolute', 'mean_generalized_eigenvalue'};
    assert(ismember(mode, valid), ...
        'lambda_scale_mode must be absolute or mean_generalized_eigenvalue.');
end

function scale = regularization_scale(K_h, Q_alpha, mode)
%% Return the loss-curvature scale relative to the RKHS penalty metric.

    if strcmp(mode, 'absolute')
        scale = 1;
        return
    end

    [L, flag] = chol(0.5 * (K_h + K_h.'), 'lower');
    if flag ~= 0
        error('cross_validate_lambda:ScaleNeedsSPD', ...
            'Normalized lambda scaling requires positive-definite K_h.');
    end

    B = L \ Q_alpha / L.';
    B = 0.5 * (B + B.');
    scale = regularization_scale_from_whitened(B, mode);
end

function scale = regularization_scale_from_whitened(B, mode)
%% Scale lambda by the mean generalized eigenvalue of (Q_alpha,K_h).

    if strcmp(mode, 'absolute')
        scale = 1;
        return
    end

    scale = real(trace(B)) / size(B, 1);
    assert(isfinite(scale) && scale > 0, ...
        'Mean generalized eigenvalue must be finite and strictly positive.');
end

function scale = regularization_scale_from_trace(Btrace, p, mode)
%% Return the same normalized scale when only trace(B) is available.

    if strcmp(mode, 'absolute')
        scale = 1;
        return
    end

    scale = real(Btrace) / p;
    assert(isfinite(scale) && scale > 0, ...
        'Mean generalized eigenvalue must be finite and strictly positive.');
end

function minTrain = get_min_train_dates(cfg, T)
%% Read the initial expanding-window training length.

    assert(isfield(cfg.training, 'cv') ...
        && isfield(cfg.training.cv, 'min_train_dates') ...
        && ~isempty(cfg.training.cv.min_train_dates), ...
        'Set cfg.training.cv.min_train_dates explicitly.');

    minTrain = cfg.training.cv.min_train_dates;

    assert(isscalar(minTrain) && minTrain == floor(minTrain), ...
        'min_train_dates must be an integer scalar.');
    assert(minTrain >= 2, 'min_train_dates must be at least 2.');
    assert(minTrain < T, ...
        'min_train_dates (%d) must be smaller than T (%d).', minTrain, T);
end

function solver = get_solver(cfg)
%% Read the numerical solver used for the one-step CV path.

    solver = 'direct';
    if isfield(cfg.training, 'cv') && isfield(cfg.training.cv, 'solver') ...
            && ~isempty(cfg.training.cv.solver)
        solver = char(lower(string(cfg.training.cv.solver)));
    end
end

function refreshEvery = get_woodbury_refresh(cfg)
%% Read the numerical EVD refresh interval for the Woodbury CV solver.

    assert(isfield(cfg.training, 'cv') ...
        && isfield(cfg.training.cv, 'woodbury_refresh') ...
        && ~isempty(cfg.training.cv.woodbury_refresh), ...
        'Set cfg.training.cv.woodbury_refresh explicitly.');

    refreshEvery = cfg.training.cv.woodbury_refresh;

    assert(isscalar(refreshEvery) && refreshEvery == floor(refreshEvery) ...
        && refreshEvery >= 1, ...
        'woodbury_refresh must be a positive integer.');
end

function tf = use_parallel(cfg)
%% Read whether direct lambda candidates may be evaluated in parallel.

    tf = false;
    if isfield(cfg, 'parallel') && isfield(cfg.parallel, 'use')
        tf = cfg.parallel.use;
    end
    if isfield(cfg.training, 'cv') && isfield(cfg.training.cv, 'parallel')
        tf = tf && cfg.training.cv.parallel;
    end
end
