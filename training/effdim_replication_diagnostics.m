function tbl = effdim_replication_diagnostics(sel, nRows)
%% Per-replication effective-dimension diagnostics as table columns.
%
% sel is the output of select_lambda_effective_dimension for one simulated
% dataset. The returned nRows-by-9 table repeats the same values on every
% shape-test row of that replication so it can be concatenated
% horizontally with the replication's result rows.
%
% effdim_whitened_cond is the 2-norm condition number of the K_h-whitened
% system W + lambda*I (W = R^{-T} Q_alpha R^{-1}, K_h = R'R), computed from
% the eigenvalues the selector already has (squared singular values of the
% payoff factor times R^{-1}; W itself is never formed); with rcond_K_h it
% describes the conditioning of the fit at small lambda without another
% factorization.

mu = sel.eigenvalues(:);
lambda = sel.lambda;
whitenedCond = (max(mu) + lambda) / (min(mu) + lambda);

one = table(sel.target, sel.N_at_lambda, sel.residual, ...
    sel.numerical_rank, sel.rank_upper_bound, whitenedCond, sel.rcond_K_h, ...
    sel.discarded_eigenvalue_ratio, string(sel.solver.name), ...
    'VariableNames', {'effdim_target', 'effdim_N_at_lambda', ...
    'effdim_residual', 'effdim_numerical_rank', 'effdim_rank_upper_bound', ...
    'effdim_whitened_cond', 'effdim_rcond_K_h', ...
    'effdim_discarded_eigenvalue_ratio', 'effdim_solver'});
tbl = repmat(one, nRows, 1);

end
