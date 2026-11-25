function jobLambda = resolve_job_lambda(lambda, jobSeed)
%% Resolve a study lambda argument into one lambda per replication job.
%
% lambda may be
%   []                        cross-validate inside each replication;
%   scalar                    one fixed lambda for every replication;
%   N-by-2 [seed lambda_fit]  the saved per-replication fitting lambda, matched
%                             by seed. This is the deterministic fixed-lambda
%                             replay: no CV is run, and the existing fixed
%                             branch of the study then sets
%                             lambda_inference = lambda_fit.
%
% Returns a numel(jobSeed)-by-1 cell array so that parfor can slice it. Each
% element is either [] (cross-validate) or a positive finite scalar. Nothing
% about the estimator, simulator, shape test, HAC, kernel or output schema is
% affected; only which lambda each job receives.

jobSeed = double(jobSeed(:));
nJobs = numel(jobSeed);

if isempty(lambda)
    jobLambda = repmat({[]}, nJobs, 1);
    return
end

lambda = double(lambda);

if isscalar(lambda)
    assert(isfinite(lambda) && lambda > 0, ...
        'resolve_job_lambda:BadScalar', ...
        'A fixed lambda must be positive and finite.');
    jobLambda = repmat({lambda}, nJobs, 1);
    return
end

assert(ismatrix(lambda) && ~isempty(lambda) && size(lambda, 2) == 2, ...
    'resolve_job_lambda:BadLookup', ...
    'A lambda lookup must be a non-empty N-by-2 [seed, lambda_fit] matrix.');

lookupSeed = lambda(:,1);
lookupLambda = lambda(:,2);

assert(all(isfinite(lookupSeed)), ...
    'resolve_job_lambda:BadLookupSeed', 'Lookup seeds must be finite.');

uniqueSeed = unique(lookupSeed);
assert(numel(uniqueSeed) == numel(lookupSeed), ...
    'resolve_job_lambda:DuplicateSeed', ...
    'The lambda lookup contains %d duplicate seed entries.', ...
    numel(lookupSeed) - numel(uniqueSeed));

assert(all(isfinite(lookupLambda) & lookupLambda > 0), ...
    'resolve_job_lambda:BadLookupLambda', ...
    'Every lambda_fit in the lookup must be positive and finite.');

[found, idx] = ismember(jobSeed, lookupSeed);
assert(all(found), 'resolve_job_lambda:MissingSeed', ...
    '%d of %d job seeds are absent from the lambda lookup (first missing seed: %d).', ...
    sum(~found), nJobs, jobSeed(find(~found, 1)));

jobLambda = num2cell(lookupLambda(idx));

end
