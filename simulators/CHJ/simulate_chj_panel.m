function [df, truth] = simulate_chj_panel(cfg)
%% Generate a one-step CHJ option panel.
%
% Physical dynamics:
%   R_t = exp(r + (mu - 1/2) h_t + sqrt(h_t) z_t)
%   h_{t+1} = omega + beta h_t + alpha (z_t - gamma sqrt(h_t))^2
%
% SDF used for pricing options:
%   M_t = R_t^phi * exp(delta + eta h_t + xi (h_{t+1} - h_t)).
%
% Option forward prices are computed directly as E_P[M payoff] / E_P[M].
% VIX is the CHJ annualized volatility state: sqrt(252*h_t).

    if cfg.sim.horizon_days ~= 1
        error('simulate_chj_panel:BadHorizon', ...
            'The CHJ simulator is implemented for horizon_days = 1.');
    end

    T = cfg.sim.T;
    J = cfg.sim.J;

    S0 = cfg.chj.S0;
    h0 = cfg.chj.h0;
    mu = cfg.chj.mu;
    r = cfg.chj.r;
    omega = cfg.chj.omega;
    beta = cfg.chj.beta;
    alpha = cfg.chj.alpha;
    gamma = cfg.chj.gamma;
    xi = cfg.chj.xi;

    params = calculate_chj_parameters(mu, xi, omega, beta, alpha, gamma, r);

    mGrid = chj_option_moneyness_grid(cfg);
    wGrid = trapz_weights_local(mGrid);

    [zGH, wGH] = gauss_hermite_standard_normal(81);

    if isfield(cfg.sim, 'seed') && ~isempty(cfg.sim.seed)
        rng(cfg.sim.seed);
    end

    S = zeros(T + 1, 1);
    h = zeros(T + 1, 1);
    z = zeros(T, 1);

    S(1) = S0;
    h(1) = h0;

    R = zeros(T, 1);
    Delta = zeros(T, 1);
    VIX = zeros(T, 1);
    variance = zeros(T, 1);
    S_start = zeros(T, 1);
    S_end = zeros(T, 1);

    option_price = zeros(T, J);
    option_forward_price = zeros(T, J);
    rawPayoff = zeros(T, J);
    netPayoff = zeros(T, J);
    isCall = repmat(mGrid >= 1, 1, T).';

    bond = exp(-r);

    for t = 1:T
        h_t = h(t);
        S_t = S(t);

        z(t) = randn;

        R(t) = exp(r + (mu - 0.5) * h_t + sqrt(h_t) * z(t));
        S(t + 1) = S_t * R(t);

        h(t + 1) = omega + beta * h_t ...
            + alpha * (z(t) - gamma * sqrt(h_t))^2;

        if h(t + 1) <= 0 || ~isfinite(h(t + 1))
            error('simulate_chj_panel:BadVariance', ...
                'Invalid variance at t=%d.', t);
        end

        Delta(t) = R(t) - exp(r);
        VIX(t) = sqrt(252 * h_t);
        variance(t) = h_t;
        S_start(t) = S_t;
        S_end(t) = S(t + 1);

        Rq = exp(r + (mu - 0.5) * h_t + sqrt(h_t) * zGH);
        h_next_q = omega + beta * h_t ...
            + alpha * (zGH - gamma * sqrt(h_t)).^2;

        Mq = Rq.^params.phi .* exp(params.delta + params.eta * h_t ...
            + xi * (h_next_q - h_t));

        EM = sum(wGH .* Mq);

        for j = 1:J
            m = mGrid(j);

            if m >= 1
                payoff_q = max(Rq - m, 0);
                rawPayoff(t, j) = max(R(t) - m, 0);
            else
                payoff_q = max(m - Rq, 0);
                rawPayoff(t, j) = max(m - R(t), 0);
            end

            option_forward_price(t, j) = sum(wGH .* Mq .* payoff_q) / EM;
            option_price(t, j) = S_t * bond * option_forward_price(t, j);
            netPayoff(t, j) = rawPayoff(t, j) - option_forward_price(t, j);
        end
    end

    dateId = repelem((0:T-1).', J);
    mLong = repmat(mGrid, T, 1);

    df = table();
    df.date = dateId;
    df.frwd_moneyness = mLong;
    df.SP500_ret = repelem(Delta, J);
    df.weight = repmat(wGrid, T, 1);
    df.OPT_NET_R_PAYOFF = reshape(netPayoff.', [], 1);
    df.VIX = repelem(VIX, J);

    % Diagnostics; not required by the training code.
    df.R_gross = repelem(R, J);
    df.S = repelem(S_start, J);
    df.S_next = repelem(S_end, J);
    df.variance = repelem(variance, J);
    df.option_price = reshape(option_price.', [], 1);
    df.option_forward_price = reshape(option_forward_price.', [], 1);
    df.raw_option_payoff = reshape(rawPayoff.', [], 1);
    df.is_call = reshape(isCall.', [], 1);
    df.is_put = ~df.is_call;

    truth = struct();
    truth.S = S;
    truth.h = h;
    truth.z = z;
    truth.R = R;
    truth.Delta = Delta;
    truth.VIX = VIX;
    truth.m_grid = mGrid;
    truth.weights = wGrid;
    truth.option_price = option_price;
    truth.option_forward_price = option_forward_price;
    truth.raw_payoff = rawPayoff;
    truth.net_payoff = netPayoff;
    truth.chj_params = params;
    truth.horizon_days = cfg.sim.horizon_days;
end

function [zGH, wGH] = gauss_hermite_standard_normal(nGH)
% Nodes and weights for E[f(Z)], Z ~ N(0,1).

    ii = (1:nGH-1).';
    C = diag(sqrt(ii/2), 1) + diag(sqrt(ii/2), -1);
    [V, D] = eig(C);
    gh_x = diag(D);
    gh_w = V(1,:).^2;

    [gh_x, idx] = sort(gh_x);
    gh_w = gh_w(idx).';

    zGH = sqrt(2) * gh_x;
    wGH = gh_w;
end

function w = trapz_weights_local(x)
% Composite trapezoidal weights on a one-dimensional grid.

    n = numel(x);
    if n < 2 || any(diff(x) <= 0)
        error('trapz_weights_local:BadGrid', ...
            'The moneyness grid must be strictly increasing.');
    end

    w = zeros(n, 1);
    w(1) = 0.5 * (x(2) - x(1));
    w(n) = 0.5 * (x(n) - x(n-1));

    if n > 2
        w(2:n-1) = 0.5 * (x(3:n) - x(1:n-2));
    end
end
