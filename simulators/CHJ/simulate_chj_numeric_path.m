function path = simulate_chj_numeric_path(cfg, seed)
%% Simulate one CHJ return, variance, and VIX path without option tables.
%
% The physical dynamics are identical to simulate_chj_panel and the
% numeric alpha builder. This function separates the stochastic path draw from
% deterministic option pricing so each replication is simulated exactly once.
% VIX is the CHJ annualized volatility state: sqrt(252*h_t).

if nargin < 2 || isempty(seed)
    seed = cfg.sim.seed;
end

assert(cfg.sim.horizon_days == 1, ...
    'simulate_chj_numeric_path supports the one-day CHJ design only.');

T = cfg.sim.T;
chj = cfg.chj;

rng(seed, 'twister');

S = zeros(T + 1, 1);
h = zeros(T + 1, 1);
z = zeros(T, 1);
R = zeros(T, 1);
Delta = zeros(T, 1);
VIX = zeros(T, 1);

S(1) = chj.S0;
h(1) = chj.h0;

for t = 1:T
    h_t = h(t);
    z(t) = randn;

    R(t) = exp(chj.r + (chj.mu - 0.5) * h_t + sqrt(h_t) * z(t));
    S(t + 1) = S(t) * R(t);

    h(t + 1) = chj.omega + chj.beta * h_t ...
        + chj.alpha * (z(t) - chj.gamma * sqrt(h_t))^2;

    if h(t + 1) <= 0 || ~isfinite(h(t + 1))
        error('simulate_chj_numeric_path:BadVariance', ...
            'Invalid variance at t=%d.', t);
    end
    if S(t + 1) <= 0 || ~isfinite(S(t + 1))
        error('simulate_chj_numeric_path:BadPrice', ...
            'Invalid stock price at t=%d.', t);
    end

    Delta(t) = R(t) - exp(chj.r);
    VIX(t) = sqrt(252 * h_t);
end

path = struct();
path.seed = seed;
path.T = T;
path.S = S;
path.h = h;
path.z = z;
path.R = R;
path.Delta = Delta;
path.VIX = VIX;

end
