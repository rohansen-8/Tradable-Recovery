function [lsEndo, lsExo] = option_level_median_lengthscales(cfg, df)
%% Return the option-level median-pairwise-distance kernel lengthscales.
%
% Both median heuristics are built from the option-level estimation rows
%       endogenous: frwd_moneyness, one value per option
%       exogenous : VIX, repeated within each date
% and NOT from the basis grids. The distinction is large: on the monthly panel
% the row-based rule gives ls_endo ~ 0.070 against ~0.186 for the grid.
%
% In one dimension the cityblock and euclidean metrics coincide, so one median
% pairwise distance serves both kernels.

names = df.Properties.VariableNames;
mName = pick_column(names, {'frwd_moneyness','m'});
lsEndo = median_pairwise_lengthscale(df.(mName));

lsExo = [];
if cfg.has_exo
    exoCols = cellstr(cfg.exo_cols);
    lsExo = zeros(1, numel(exoCols));
    for j = 1:numel(exoCols)
        lsExo(j) = median_pairwise_lengthscale(df.(exoCols{j}));
    end
end

end

function name = pick_column(names, candidates)
%% Return the first matching panel column name.

for j = 1:numel(candidates)
    if any(strcmp(names, candidates{j}))
        name = candidates{j};
        return
    end
end
error('option_level_median_lengthscales:MissingColumn', ...
    'Missing one of these columns: %s.', strjoin(candidates, ', '));

end
