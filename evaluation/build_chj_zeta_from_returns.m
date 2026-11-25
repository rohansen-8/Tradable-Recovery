function zeta = build_chj_zeta_from_returns(R, h_t, pre)
%% CHJ moneyness payoff loadings zeta_t(R) for one date at chosen gross returns.
%
% Reproduces the date-t moneyness block of build_chj_alpha_numeric for an
% arbitrary vector of gross returns R: the forward position (R - e^r), and
% the strike-weighted options net of their date-t forward prices, both priced
% under the conditional variance h_t. With R = path.R(t) this equals
% info.zeta(:, t) of build_chj_alpha_numeric (checked in
% tests/test_chj_hj_recovery.m). The full payoff vector is
% alpha_t(R) = kron(kV_t, zeta_t(R)).
%
% Returns an (n+1)-by-numel(R) matrix.

R = R(:);
assert(isscalar(h_t) && isfinite(h_t) && h_t > 0, ...
    'build_chj_zeta_from_returns:BadVariance', 'h_t must be a positive scalar.');

chj = pre.chj;
params = pre.params;
mGrid = pre.option_m_grid(:);
wGrid = pre.option_weights(:);
gh_z = pre.gh_z(:);
gh_w = pre.gh_w(:);
isPut = mGrid < 1;

% Date-t forward option prices (same quadrature as the estimation panel).
Rq = exp(chj.r + (chj.mu - 0.5) * h_t + sqrt(h_t) * gh_z);
h_next_q = chj.omega + chj.beta * h_t ...
    + chj.alpha * (gh_z - chj.gamma * sqrt(h_t)).^2;
Mq = Rq.^params.phi .* exp(params.delta + params.eta * h_t ...
    + chj.xi * (h_next_q - h_t));
EM = sum(gh_w .* Mq);
payoffQ = max(Rq - mGrid.', 0);
payoffQ(:, isPut) = max(mGrid(isPut).' - Rq, 0);
forwardPrice = ((gh_w .* Mq).' * payoffQ / EM).';   % J-by-1

% Realized payoffs at the requested returns.
rawPayoff = max(R.' - mGrid, 0);                     % J-by-K
rawPayoff(isPut, :) = max(mGrid(isPut) - R.', 0);
payoff = wGrid .* (rawPayoff - forwardPrice);
Delta = R.' - exp(chj.r);                            % 1-by-K

zeta = zeros(numel(pre.mon_grid) + 1, numel(R));
zeta(1, :) = pre.k00 * Delta + pre.k0m * payoff;
zeta(2:end, :) = pre.kg0 * Delta + pre.Kgm * payoff;

end
