function pop = build_bs_population_moments_exact(cfg, mon_grid, pre)
%% Build exact BS population moments for the restricted payoff representer.
%
% The primitive return vector is
%   x_t = [R_t-1; w_j (O_j(R_t)-q_j)]_{j=1}^J,
% where q_j is the exact Black--Scholes forward option price under Q.
% Because R_t is lognormal under P and each OTM option payoff is piecewise
% linear in R_t, E[x_t] and E[x_t x_t'] are available analytically from
% truncated lognormal moments. This avoids Gauss--Hermite error at option
% payoff kinks and keeps the P moments exactly consistent with the Q prices.
%
% The restricted representer satisfies alpha_t = B x_t, hence
%   alpha_bar = B E[x_t],
%   Q_alpha   = B E[x_t x_t'] B'.

if nargin < 2 || isempty(mon_grid)
    mon_grid = linspace(cfg.sim.m_min, cfg.sim.m_max, cfg.training.n_m_grid).';
end
if nargin < 3 || isempty(pre)
    pre = build_bs_alpha_precompute(cfg, mon_grid);
end

m = pre.mGrid(:);
w = pre.w(:);
q = pre.q(:);
J = numel(m);

muLog = (cfg.bs.mu - 0.5 * cfg.bs.sigma^2) * cfg.bs.dt;
sLog = cfg.bs.sigma * sqrt(cfg.bs.dt);

ER = full_moment(muLog, sLog, 1);
ER2 = full_moment(muLog, sLog, 2);

EO = zeros(J, 1);
ERO = zeros(J, 1);
EOO = zeros(J, J);

for j = 1:J
    K = m(j);
    if K <= 1
        p0 = lower_moment(K, muLog, sLog, 0);
        p1 = lower_moment(K, muLog, sLog, 1);
        p2 = lower_moment(K, muLog, sLog, 2);
        EO(j) = K * p0 - p1;
        ERO(j) = K * p1 - p2;
    else
        p0 = upper_moment(K, muLog, sLog, 0);
        p1 = upper_moment(K, muLog, sLog, 1);
        p2 = upper_moment(K, muLog, sLog, 2);
        EO(j) = p1 - K * p0;
        ERO(j) = p2 - K * p1;
    end
end

for i = 1:J
    Ki = m(i);
    for j = i:J
        Kj = m(j);
        if Ki <= 1 && Kj <= 1
            Kcut = min(Ki, Kj);
            p0 = lower_moment(Kcut, muLog, sLog, 0);
            p1 = lower_moment(Kcut, muLog, sLog, 1);
            p2 = lower_moment(Kcut, muLog, sLog, 2);
            value = Ki * Kj * p0 - (Ki + Kj) * p1 + p2;
        elseif Ki > 1 && Kj > 1
            Kcut = max(Ki, Kj);
            p0 = upper_moment(Kcut, muLog, sLog, 0);
            p1 = upper_moment(Kcut, muLog, sLog, 1);
            p2 = upper_moment(Kcut, muLog, sLog, 2);
            value = p2 - (Ki + Kj) * p1 + Ki * Kj * p0;
        else
            % OTM puts have support below one and OTM calls above one.
            value = 0;
        end
        EOO(i, j) = value;
        EOO(j, i) = value;
    end
end

mean_x = zeros(J + 1, 1);
mean_x(1) = ER - 1;
mean_x(2:end) = w .* (EO - q);

Q_x = zeros(J + 1, J + 1);
Q_x(1, 1) = ER2 - 2 * ER + 1;

crossDeltaOption = ERO - EO - q * (ER - 1);
Q_x(1, 2:end) = (w .* crossDeltaOption).';
Q_x(2:end, 1) = Q_x(1, 2:end).';

netOptionMoment = EOO ...
    - EO * q.' ...
    - q * EO.' ...
    + q * q.';
Q_x(2:end, 2:end) = (w * w.') .* netOptionMoment;
Q_x = 0.5 * (Q_x + Q_x.');

B = [pre.k00, pre.k0m; pre.g0, pre.Kmm];
alpha_bar = B * mean_x;
Q_alpha = B * Q_x * B.';
Q_alpha = 0.5 * (Q_alpha + Q_alpha.');

pop = struct();
pop.alpha_bar = alpha_bar;
pop.Q_alpha = Q_alpha;
pop.mean_x = mean_x;
pop.Q_x = Q_x;
pop.B = B;
pop.ER = ER;
pop.ER2 = ER2;
pop.EO = EO;
pop.ERO = ERO;
pop.EOO = EOO;

end

function value = full_moment(muLog, sLog, p)
%% Return E[R^p] for log R ~ N(muLog,sLog^2).

value = exp(p * muLog + 0.5 * p^2 * sLog^2);

end

function value = lower_moment(K, muLog, sLog, p)
%% Return E[R^p 1{R<=K}] for a lognormal R.

z = (log(K) - muLog - p * sLog^2) / sLog;
value = full_moment(muLog, sLog, p) * normcdf(z);

end

function value = upper_moment(K, muLog, sLog, p)
%% Return E[R^p 1{R>K}] for a lognormal R.

value = full_moment(muLog, sLog, p) - lower_moment(K, muLog, sLog, p);

end
