function out = matern72_dyy(x, y, theta)
%% Evaluate the second derivative of the Matern 7/2 kernel with respect to y.

    out = -matern72_dxy(x, y, theta);
    
end
