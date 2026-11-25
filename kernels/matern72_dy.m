function out = matern72_dy(x, y, theta)
%% Evaluate the first derivative of the Matern 7/2 kernel with respect to y.
    z = x - y;
    r = abs(z) ./ theta;

    % Off-diagonal formula
    out = (1./(5*theta.^2)) .* exp(-r) .* z .* (1 + r + (1/3).*r.^2);

    % Diagonal correction for x = y
    tol = 1e-12;
    mask = r < tol;
    out(mask) = 0;
end
