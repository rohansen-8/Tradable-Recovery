function out = matern72_dxyy(x, y, theta)
%% Evaluate the mixed third derivative of the Matern 7/2 kernel.
    z = x - y;
    r = abs(z) ./ theta;

    % Off-diagonal formula
    out = -(1./(5*theta.^4)) .* exp(-r) .* z .* ((1/3).*r.^2 - r - 1);

    % Diagonal correction for x = y
    tol = 1e-12;
    mask = r < tol;
    out(mask) = 0;
    
end
