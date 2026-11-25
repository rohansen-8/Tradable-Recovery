function out = laplace_k(x, y, theta)
%% Evaluate the Laplace kernel.
    z = x - y;
    r = abs(z) ./ theta;
    c = sqrt(pi / 2);
    %c = 1;
    out = c * exp(-r);
end
