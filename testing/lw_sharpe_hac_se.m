function out = lw_sharpe_hac_se(r)
%% Ledoit-Wolf (2008) HAC standard error of a Sharpe-ratio difference.
%
%   out = lw_sharpe_hac_se(r)     r: T-by-2 paired per-period returns
%                                    (column 1 strategy, column 2 benchmark)
%
% delta = SR1 - SR2 per period, SR = mean/std (sample SD, T-1). The
% standard error is the delta method on the four moments
% (mu1, mu2, E r1^2, E r2^2) with a prewhitened Quadratic Spectral HAC
% (Andrews 1991; Andrews-Monahan 1992):
%   - VAR(1) prewhitening without intercept on the demeaned moment series;
%   - Andrews AR(1) plug-in bandwidth, AR(1) fitted with an intercept to
%     each prewhitened component, unit weights, sample size T-1;
%   - QS weights over all lags, small-sample factor (T-1)/(T-1-4);
%   - gradient rescaled by sqrt((T-1)/T) so that the statistic refers to
%     the sample-SD Sharpe difference.

assert(ismatrix(r) && size(r, 2) == 2 && size(r, 1) > 10 && all(isfinite(r(:))), ...
    'lw_sharpe_hac_se:BadInput', 'r must be a finite T-by-2 matrix with T > 10.');
T = size(r, 1);
mu = mean(r, 1);
gm = mean(r.^2, 1);
delta = mu(1) / std(r(:, 1)) - mu(2) / std(r(:, 2));

Y = [r(:, 1) - mu(1), r(:, 2) - mu(2), r(:, 1).^2 - gm(1), r(:, 2).^2 - gm(2)];
Psi = local_prewhitened_qs(Y);
g = lw_sharpe_gradient(mu, gm) * sqrt((T - 1) / T);
se = sqrt(g * Psi * g.' / T);

out = struct('T', T, 'delta', delta, 'se', se, 'd', delta / se, 'Psi', Psi, ...
    'sr', [mu(1) / std(r(:, 1)), mu(2) / std(r(:, 2))]);
end

function G = local_prewhitened_qs(Y)
[T, k] = size(Y);
X = Y(1:end-1, :);
Z = Y(2:end, :);
A = (X \ Z).';                       % VAR(1), no intercept
V = Z - X * A.';
Tn = T - 1;

rho = zeros(1, k);
s2 = zeros(1, k);
for j = 1:k
    xa = [ones(Tn - 1, 1), V(1:end-1, j)];
    c = xa \ V(2:end, j);
    rho(j) = c(2);
    e = V(2:end, j) - xa * c;
    s2(j) = mean(e.^2);
end
a2 = sum(4 * rho.^2 .* s2.^2 ./ (1 - rho).^8) / sum(s2.^2 ./ (1 - rho).^4);
S = 1.3221 * (a2 * Tn)^0.2;

G = (V.' * V) / Tn;
for j = 1:Tn-1
    Gj = (V(j+1:end, :).' * V(1:end-j, :)) / Tn;
    G = G + local_qs(j / S) * (Gj + Gj.');
end
G = G * Tn / (Tn - k);
D = (eye(k) - A) \ eye(k);
G = D * G * D.';
G = 0.5 * (G + G.');
end

function w = local_qs(x)
z = 6 * pi * x / 5;
w = 25 / (12 * pi^2 * x^2) * (sin(z) / z - cos(z));
end
