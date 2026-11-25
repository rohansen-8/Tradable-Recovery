function params = calculate_chj_parameters(mu, xi, omega, beta, alpha, gamma, r)
%% CHJ pricing-kernel parameters from daily physical parameters.

    if beta + alpha * gamma^2 >= 1
        error('calculate_chj_parameters:Stationarity', ...
            'Stationarity requires beta + alpha * gamma^2 < 1.');
    end
    if 1 - 2 * alpha * xi <= 0
        error('calculate_chj_parameters:BadXi', ...
            'The CHJ transform requires 1 - 2 * alpha * xi > 0.');
    end

    d = 1 - 2 * alpha * xi;

    phi = -(mu - 0.5 + gamma) * d + gamma - 0.5;
    delta = -(phi + 1) * r - xi * omega + 0.5 * log(d);
    eta = -(mu - 0.5) * phi - xi * alpha * gamma^2 + (1 - beta) * xi ...
        - (phi - 2 * xi * alpha * gamma)^2 / (2 * d);

    params = struct();
    params.phi = phi;
    params.delta = delta;
    params.eta = eta;
end
