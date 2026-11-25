function payoff = build_payoff_factor(alpha)
%% Triangular factor of the payoff panel, shared by lambda selection and the fit.
%
% For alpha (Nh-by-T, one payoff representer per date) form the economy QR
% factorization of the T-by-(Nh+1) matrix [alpha', 1] / sqrt(T) and keep its
% triangular factor R = [F, b]. Because the orthogonal factor drops out,
%     F' F = alpha alpha' / T = Q_alpha     and     F' b = alpha 1 / T = alpha_bar,
% so F is a reduced representation of the payoff Gram operator and b carries
% the pricing target. Neither Q_alpha nor any Gram matrix is formed, so
% directions with small payoff variation keep their accuracy instead of being
% squared into rounding noise.
%
% Output payoff: F (r-by-Nh, r = min(T, Nh+1)), b (r-by-1), T, Nh.

assert(ismatrix(alpha) && ~isempty(alpha) && all(isfinite(alpha(:))), ...
    'build_payoff_factor:BadAlpha', 'alpha must be a finite Nh-by-T matrix.');
[Nh, T] = size(alpha);
[~, R] = qr([alpha.', ones(T, 1)] / sqrt(T), 0);

payoff = struct();
payoff.F = R(:, 1:Nh);
payoff.b = R(:, Nh + 1);
payoff.T = T;
payoff.Nh = Nh;

end
