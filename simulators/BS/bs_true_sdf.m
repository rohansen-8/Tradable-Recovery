function M = bs_true_sdf(R, mu, sigma, dt, r)
%% Evaluate the Black--Scholes likelihood-ratio SDF dQ/dP.
%
%   M = bs_true_sdf(R, mu, sigma, dt, r)
%
%   Under P:
%       log R = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*eps,
%       eps ~ N(0,1).
%   Under Q:
%       log R = (r  - 0.5*sigma^2)*dt + sigma*sqrt(dt)*eps_Q.
%   The SDF is the Radon--Nikodym derivative dQ/dP.

    if nargin < 5 || isempty(r)
        r = 0;
    end
    if sigma <= 0
        error('sigma must be positive.');
    end
    if dt <= 0
        error('dt must be positive.');
    end
    if any(R(:) <= 0)
        error('All gross returns R must be positive.');
    end

    theta = (mu - r) / sigma;
    epsP = (log(R) - (mu - 0.5 * sigma^2) * dt) ./ (sigma * sqrt(dt));

    M = exp(-theta * sqrt(dt) .* epsP - 0.5 * theta^2 * dt);
end
