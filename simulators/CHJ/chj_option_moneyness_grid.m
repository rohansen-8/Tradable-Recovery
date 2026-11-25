function mGrid = chj_option_moneyness_grid(cfg)
%% Return the fixed CHJ option grid in forward moneyness.

assert(isfield(cfg.sim, 'option_grid_type') ...
    && strcmp(cfg.sim.option_grid_type, 'forward_moneyness_linspace'), ...
    'CHJ option grid must be equally spaced directly in forward moneyness.');

mGrid = linspace(cfg.sim.m_min, cfg.sim.m_max, cfg.sim.J).';

end
