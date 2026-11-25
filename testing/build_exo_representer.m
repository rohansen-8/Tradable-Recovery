function exo = build_exo_representer(cfg, model_state, df, opts)
%% Build the exogenous representer loading e(mu_V).
%
% Equal-date averaging uses one exogenous observation per date with weight
% 1/T. Equal-date support
% is clipped to the fitted exogenous-grid range before kernel averaging.

if nargin < 3
    df = [];
end
if nargin < 4 || isempty(opts)
    opts = struct();
end

if ~cfg.has_exo
    exo = struct('mode', 'none', 'e', 1, 'support', [], 'weights', 1, ...
        'support_raw', [], 'clip_bounds', [], 'n_clipped', 0, 'frac_clipped', 0);
    return;
end

mode = get_mode(cfg, opts);
exo_grid = as_matrix(model_state.exo_grid);

supportRaw = [];
clipBounds = [];
nClipped = 0;
fracClipped = 0;

switch lower(mode)
    case 'conditional'
        if ~isfield(opts, 'value') || isempty(opts.value)
            error('build_exo_representer:MissingValue', ...
                'Conditional tests require opts.value.');
        end
        support = reshape(opts.value, 1, []);
        weights = 1;

    case 'equal_date'
        if isfield(opts, 'observations') && ~isempty(opts.observations)
            obs = as_matrix(opts.observations);
        else
            obs = date_exo_observations(cfg, df);
        end

        supportRaw = obs;
        clipToGrid = get_equal_date_clip_flag(cfg, opts);
        if clipToGrid
            [obs, clipBounds, nClipped] = clip_to_exo_grid(obs, exo_grid);
            fracClipped = nClipped / size(obs, 1);
        end

        support = obs;
        weights = ones(size(obs, 1), 1) / size(obs, 1);

    case 'bucketed'
        if isfield(opts, 'support') && isfield(opts, 'weights')
            support = as_matrix(opts.support);
            weights = opts.weights(:);
        else
            obs = date_exo_observations(cfg, df);
            B = get_field(opts, 'B', 5);
            [support, weights] = bucket_exo(obs, B);
        end

    case 'region_specific'
        if ~isfield(opts, 'bounds') || isempty(opts.bounds)
            error('build_exo_representer:MissingBounds', ...
                'Region-specific tests require opts.bounds = [m_min, m_max].');
        end
        [support, weights] = region_exo_weights(cfg, df, opts.bounds);

    otherwise
        error('build_exo_representer:BadMode', ...
            'Unsupported exogenous mode: %s.', mode);
end

weights = weights(:);
weights = weights / sum(weights);

exo = struct();
exo.mode = mode;
exo.e = exo_kernel_matrix(exo_grid, support, cfg, model_state) * weights;
exo.support = support;
exo.weights = weights;
exo.support_raw = supportRaw;
exo.clip_bounds = clipBounds;
exo.n_clipped = nClipped;
exo.frac_clipped = fracClipped;

end

function mode = get_mode(cfg, opts)
%% Resolve the requested exogenous averaging mode.

if isfield(opts, 'mode') && ~isempty(opts.mode)
    mode = char(opts.mode);
elseif isfield(cfg.testing, 'exo') && isfield(cfg.testing.exo, 'defaultMode')
    mode = cfg.testing.exo.defaultMode;
elseif isfield(cfg.testing, 'exo') && isfield(cfg.testing.exo, 'mode')
    mode = cfg.testing.exo.mode;
else
    mode = 'conditional';
end

end

function tf = get_equal_date_clip_flag(cfg, opts)
%% Resolve whether equal-date observations are clipped to fitted grid support.

if isfield(opts, 'clip_to_grid') && ~isempty(opts.clip_to_grid)
    tf = logical(opts.clip_to_grid);
elseif isfield(cfg.testing, 'exo') && isfield(cfg.testing.exo, 'clip_to_grid') ...
        && ~isempty(cfg.testing.exo.clip_to_grid)
    tf = logical(cfg.testing.exo.clip_to_grid);
else
    % Default: clip for both the figures and equal-date shape evaluation.
    tf = true;
end

end

function [obsClipped, bounds, nClipped] = clip_to_exo_grid(obs, exo_grid)
%% Clip each exogenous coordinate to the fitted grid's coordinate-wise range.

obs = as_matrix(obs);
exo_grid = as_matrix(exo_grid);
assert(size(obs, 2) == size(exo_grid, 2), ...
    'Observed exogenous dimension must match the fitted exogenous grid.');

lo = min(exo_grid, [], 1);
hi = max(exo_grid, [], 1);
assert(all(isfinite(lo)) && all(isfinite(hi)) && all(lo < hi), ...
    'Fitted exogenous grid must have a finite nondegenerate range.');

obsClipped = min(max(obs, lo), hi);
rowChanged = any(obsClipped ~= obs, 2);
nClipped = sum(rowChanged);
bounds = [lo; hi];

end

function obs = date_exo_observations(cfg, df)
%% Return one observed exogenous-state row per date.

if isempty(df)
    df = readtable(cfg.dataset.dfFile);
end
names = df.Properties.VariableNames;
timeName = pick_name(names, {'t', 'date'});
[~, firstIdx] = unique(df.(timeName), 'stable');
obs = df{firstIdx, cfg.exo_cols};
obs = as_matrix(obs);

end

function [support, weights] = bucket_exo(obs, B)
%% Construct equal-frequency exogenous buckets.

if size(obs, 2) ~= 1
    error('build_exo_representer:BucketDim', ...
        'Default bucketed averaging is defined for one exogenous variable.');
end

edges = quantile(obs, linspace(0, 1, B + 1));
edges(1) = -Inf;
edges(end) = Inf;

support = zeros(B, 1);
weights = zeros(B, 1);
for b = 1:B
    idx = obs >= edges(b) & obs <= edges(b + 1);
    if b > 1
        idx = obs > edges(b) & obs <= edges(b + 1);
    end
    support(b) = median(obs(idx));
    weights(b) = mean(idx);
end

keep = weights > 0;
support = support(keep, :);
weights = weights(keep);

end

function [support, weights] = region_exo_weights(cfg, df, bounds)
%% Weight exogenous states by option support within one moneyness region.

if isempty(df)
    df = readtable(cfg.dataset.dfFile);
end

names = df.Properties.VariableNames;
timeName = pick_name(names, {'t', 'date'});
mName = pick_name(names, {'m', 'frwd_moneyness'});

[g, firstIdx] = findgroups(df.(timeName));
nDates = max(g);
counts = zeros(nDates, 1);
for t = 1:nDates
    m_t = df.(mName)(g == t);
    counts(t) = sum(m_t >= bounds(1) & m_t <= bounds(2));
end

keep = counts > 0;
support = df{firstIdx(keep), cfg.exo_cols};
support = as_matrix(support);
weights = counts(keep) / sum(counts(keep));

end

function K = exo_kernel_matrix(X, Y, cfg, model_state)
%% Evaluate the fitted exogenous kernel matrix.

X = as_matrix(X);
Y = as_matrix(Y);

ell = get_exo_lengthscale(X, cfg, model_state);
if isscalar(ell)
    ell = repmat(ell, 1, size(X, 2));
end

switch lower(cfg.kernel.exo.type)
    case 'laplace'
        K = ones(size(X, 1), size(Y, 1));
        for j = 1:size(X, 2)
            K = K .* laplace_k(X(:, j), Y(:, j).', ell(j));
        end
    otherwise
        error('build_exo_representer:BadExoKernel', ...
            'Unsupported exogenous kernel type: %s.', cfg.kernel.exo.type);
end

end

function ell = get_exo_lengthscale(X, cfg, model_state)
%% Use the fitted exogenous lengthscale, with the common median-rule fallback.

ell = [];
if nargin >= 3 && isstruct(model_state) && isfield(model_state, 'ls_exo')
    ell = model_state.ls_exo;
end
if isempty(ell) && isfield(cfg, 'kernel') && isfield(cfg.kernel, 'exo')
    ell = cfg.kernel.exo.lengthscale;
end

if isempty(ell)
    ell = zeros(1, size(X, 2));
    for j = 1:size(X, 2)
        ell(j) = median_pairwise_lengthscale(X(:, j));
    end
end

end

function name = pick_name(names, candidates)
%% Return the first matching table variable name.

for j = 1:numel(candidates)
    if any(strcmp(names, candidates{j}))
        name = candidates{j};
        return;
    end
end
error('build_exo_representer:MissingColumn', ...
    'Missing one of these columns: %s.', strjoin(candidates, ', '));

end

function value = get_field(s, name, defaultValue)
%% Read one optional structure field.

if isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = defaultValue;
end

end

function X = as_matrix(X)
%% Convert one-dimensional vectors to column matrices.

if isempty(X)
    X = [];
elseif isvector(X)
    X = X(:);
end

end
