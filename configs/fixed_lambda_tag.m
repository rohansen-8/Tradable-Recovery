function tag = fixed_lambda_tag(lambda)
%% Deterministic folder tag for one fixed lambda: 1e-06 -> 'lambda_1e-6', 0.1 -> 'lambda_1e-1'.
% Powers of ten give 'lambda_1e<k>'; any other value uses %.6g with '.' -> 'p'.

assert(isscalar(lambda) && isfinite(lambda) && lambda > 0, 'fixed_lambda_tag:BadLambda', ...
    'lambda must be one positive finite scalar.');
k = round(log10(lambda));
if lambda == str2double(sprintf('1e%d', k))     % same double as the literal 1e<k>
    tag = sprintf('lambda_1e%d', k);
else
    tag = ['lambda_' strrep(sprintf('%.6g', lambda), '.', 'p')];
end
end
