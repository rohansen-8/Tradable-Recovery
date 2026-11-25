function grids = build_region_grids(cfg)
%% Build deterministic test grids.
%
% Supported cfg.testing.grid.type values:
%   global_midpoints : one global midpoint grid, then subset by region.
%   global_endpoints : one global endpoint grid, then subset by region.
%   region_endpoints : n_points endpoint grid inside each region.
%

    regionSpecs = cfg.testing.regionSpecs;
    bounds = vertcat(regionSpecs.bounds);

    grid_cfg = get_grid_cfg(cfg, bounds);

    if isfield(cfg, 'grids') && isfield(cfg.grids, 'path') && ~isfolder(cfg.grids.path)
        mkdir(cfg.grids.path);
    end

    grids = repmat(struct('name', '', 'label', '', 'bounds', [], 'bounds_log10', [], 'm', []), numel(regionSpecs), 1);

    if strcmpi(grid_cfg.type, 'region_endpoints')
        for j = 1:numel(regionSpecs)
            grids(j) = make_region_grid(regionSpecs(j), linspace(regionSpecs(j).bounds(1), regionSpecs(j).bounds(2), grid_cfg.n_points).');
            save_region_grid_if_needed(cfg, grids(j));
        end
        save_test_grid_if_needed(cfg, grid_cfg, vertcat(grids.m));
        return
    end

    if strcmpi(grid_cfg.type, 'global_midpoints')
        % Use the single production midpoint constructor, which avoids
        % colon-arithmetic boundary ambiguity near regional cutoffs.
        cfg_grid = cfg;
        cfg_grid.testing.grid = grid_cfg;
        m_all = build_global_shape_grid(cfg_grid);
    else
        m_all = build_global_grid(grid_cfg.domain, grid_cfg.n_points, grid_cfg.type);
    end

    for j = 1:numel(regionSpecs)
        b = regionSpecs(j).bounds;
        % Partition a common global grid without duplicating shared boundaries.
        if j < numel(regionSpecs)
            idx = (m_all >= b(1)) & (m_all < b(2));
        else
            idx = (m_all >= b(1)) & (m_all <= b(2));
        end

        if ~any(idx)
            error('build_region_grids:EmptyRegion', ...
                'No test-grid points fall in region %s.', regionSpecs(j).name);
        end

        grids(j) = make_region_grid(regionSpecs(j), m_all(idx));
        save_region_grid_if_needed(cfg, grids(j));
    end

    save_test_grid_if_needed(cfg, grid_cfg, m_all);
end

function grid = make_region_grid(regionSpec, m)
    grid = struct();
    grid.name = regionSpec.name;
    grid.label = regionSpec.label;
    grid.bounds = regionSpec.bounds;
    if isfield(regionSpec, 'bounds_log10')
        grid.bounds_log10 = regionSpec.bounds_log10;
    else
        grid.bounds_log10 = [];
    end
    grid.m = m(:);
end

function save_region_grid_if_needed(cfg, grid)
    if isfield(cfg, 'grids') && isfield(cfg.grids, 'path')
        save(fullfile(cfg.grids.path, [grid.name '.mat']), 'grid');
    end
end

function save_test_grid_if_needed(cfg, grid_cfg, m_all)
    if isfield(cfg, 'grids') && isfield(cfg.grids, 'path')
        test_grid = struct();
        test_grid.type = grid_cfg.type;
        test_grid.domain = grid_cfg.domain;
        test_grid.n_points = grid_cfg.n_points;
        test_grid.m = m_all(:);
        save(fullfile(cfg.grids.path, 'test_grid.mat'), 'test_grid');
    end
end

function grid_cfg = get_grid_cfg(cfg, bounds)
    if isfield(cfg.testing, 'grid')
        user_grid = cfg.testing.grid;
    else
        user_grid = struct();
    end

    grid_cfg = struct();

    if isfield(user_grid, 'type') && ~isempty(user_grid.type)
        grid_cfg.type = user_grid.type;
    else
        grid_cfg.type = 'global_midpoints';
    end

    if isfield(user_grid, 'domain') && ~isempty(user_grid.domain)
        grid_cfg.domain = user_grid.domain(:).';
    else
        grid_cfg.domain = [min(bounds(:, 1)), max(bounds(:, 2))];
    end

    if isfield(user_grid, 'n_points') && ~isempty(user_grid.n_points)
        grid_cfg.n_points = user_grid.n_points;
    else
        grid_cfg.n_points = 100;
    end

    if numel(grid_cfg.domain) ~= 2 || grid_cfg.domain(1) >= grid_cfg.domain(2)
        error('build_region_grids:BadDomain', 'cfg.testing.grid.domain must be [m_min, m_max].');
    end
    if grid_cfg.n_points < 1 || grid_cfg.n_points ~= round(grid_cfg.n_points)
        error('build_region_grids:BadNPoints', 'cfg.testing.grid.n_points must be a positive integer.');
    end
end

function m_all = build_global_grid(domain, n_points, type)
    switch lower(type)
        case 'global_endpoints'
            m_all = linspace(domain(1), domain(2), n_points).';
        otherwise
            error('build_region_grids:BadGridType', 'Unknown grid type %s.', type);
    end
end
