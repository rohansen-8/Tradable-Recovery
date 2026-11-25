function q = bs_forward_option_price(m, sigma, dt, optType)
%% Compute Black--Scholes forward prices of normalized options.
%
%   q = bs_forward_option_price(m, sigma, dt, optType)
%
%   Computes q = E^Q[(R-m)^+] for calls and q = E^Q[(m-R)^+] for puts,
%   where R = S_{t+1}/S_t, S_t = 1, r = 0, and
%       log R ~ N(-0.5*sigma^2*dt, sigma^2*dt) under Q.

    if sigma <= 0
        error('sigma must be positive.');
    end
    if dt <= 0
        error('dt must be positive.');
    end
    if any(m(:) <= 0)
        error('All moneyness values must be positive.');
    end

    vol = sigma * sqrt(dt);
    d1 = (-log(m) + 0.5 * sigma^2 * dt) ./ vol;
    d2 = d1 - vol;

    switch lower(optType)
        case 'call'
            q = normcdf(d1) - m .* normcdf(d2);
        case 'put'
            q = m .* normcdf(-d2) - normcdf(-d1);
        otherwise
            error('optType must be either ''call'' or ''put''.');
    end
end
