function g = lw_sharpe_gradient(mu, gm)
%% Gradient of SR1 - SR2 with respect to (mu1, mu2, E r1^2, E r2^2).
%
% SR_i = mu_i / sqrt(gm_i - mu_i^2). mu and gm may be 1-by-2 or 2-by-M
% (columns are bootstrap draws); g is 1-by-4 or 4-by-M.
if isrow(mu), mu = mu.'; gm = gm.'; end
a = (gm(1, :) - mu(1, :).^2).^1.5;
b = (gm(2, :) - mu(2, :).^2).^1.5;
g = [gm(1, :) ./ a; -gm(2, :) ./ b; -mu(1, :) ./ (2 * a); mu(2, :) ./ (2 * b)];
if size(g, 2) == 1, g = g.'; end
end
