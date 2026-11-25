function out = matern72_dxxyy(x, y, theta)
%% Evaluate the mixed fourth derivative of the Matern 7/2 kernel.
    z = x - y;
    r = abs(z) ./ theta;

    % Off-diagonal formula
    out = (1./(5*theta.^4)) .* exp(-r) .* (1 + r - 2.*r.^2 + (1/3).*r.^3);

    % Diagonal correction for x = y
    tol = 1e-12;
    mask = r < tol;
    out(mask) = 1./(5*theta.^4);
end
