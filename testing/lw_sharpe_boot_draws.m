function [delta_star, se_star] = lw_sharpe_boot_draws(r, b, M, stream)
%% Circular-block bootstrap draws of the Sharpe difference and its studentizer.
%
%   [delta_star, se_star] = lw_sharpe_boot_draws(r, b, M, stream)
%
% Each draw concatenates ceil(T/b) circular blocks of length b with
% uniformly drawn starts and truncates to T observations. delta_star is
% the sample-SD Sharpe difference of the draw. se_star is the
% Ledoit-Wolf block studentizer: with y the demeaned moment series of
% the draw and l = floor(T/b) non-overlapping blocks,
% zeta_j = sqrt(b) * (block mean of y), Psi* = sum_j zeta_j zeta_j' / l,
% se* = sqrt(g' Psi* g / T), g the gradient at the draw's moments rescaled
% by sqrt((T-1)/T). Draws are generated in chunks; stream is a RandStream.

T = size(r, 1);
L = ceil(T / b);
l = floor(T / b);
delta_star = zeros(M, 1);
se_star = zeros(M, 1);
chunk = max(1, min(M, floor(4e6 / T)));
resc = sqrt((T - 1) / T);
for m0 = 1:chunk:M
    mm = m0:min(M, m0 + chunk - 1);
    n = numel(mm);
    S = randi(stream, T, L, n);                         % block starts
    idx = mod(reshape(S, 1, L, n) - 1 + (0:b-1).', T) + 1;   % b-by-L-by-n
    idx = reshape(idx, b * L, n);
    idx = idx(1:T, :);
    x1 = r(idx(:), 1); x1 = reshape(x1, T, n);
    x2 = r(idx(:), 2); x2 = reshape(x2, T, n);
    mu = [mean(x1, 1); mean(x2, 1)];
    gm = [mean(x1.^2, 1); mean(x2.^2, 1)];
    delta_star(mm) = (mu(1, :) ./ std(x1, 0, 1) - mu(2, :) ./ std(x2, 0, 1)).';
    g = lw_sharpe_gradient(mu, gm) * resc;               % 4-by-n
    w = g(1, :) .* (x1 - mu(1, :)) + g(2, :) .* (x2 - mu(2, :)) ...
        + g(3, :) .* (x1.^2 - gm(1, :)) + g(4, :) .* (x2.^2 - gm(2, :));
    zeta = sqrt(b) * reshape(mean(reshape(w(1:l*b, :), b, l, n), 1), l, n);
    se_star(mm) = sqrt(mean(zeta.^2, 1) / T).';
end
end
