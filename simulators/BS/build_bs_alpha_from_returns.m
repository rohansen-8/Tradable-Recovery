function alpha = build_bs_alpha_from_returns(R, pre)
%% Build Black--Scholes alpha columns from a vector of gross returns.

R = R(:);
T = numel(R);
J = numel(pre.mGrid);

mGrid = pre.mGrid(:);
isCall = mGrid > 1;
isPut = ~isCall;

rawPayoff = zeros(T, J);
rawPayoff(:, isCall) = max(R - mGrid(isCall).', 0);
rawPayoff(:, isPut) = max(mGrid(isPut).' - R, 0);

netPayoff = rawPayoff - pre.q.';
payoff = netPayoff .* pre.w.';
Delta = R - 1;

alpha = zeros(pre.n + 1, T);
alpha(1, :) = pre.k00 * Delta.' + (payoff * pre.k0m.').';
alpha(2:end, :) = pre.g0 * Delta.' + pre.Kmm * payoff.';

end
