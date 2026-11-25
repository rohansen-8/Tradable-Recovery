function [levels, info] = select_vix_levels(cfg, model_state)
%% Select the five conditional VIX states by greedy furthest-point curves.
%
% 50 candidate states equally spaced
% over the fitted exogenous-grid range are each mapped to a conditional
% fitted curve on 200 evaluation points, m_raw = exp(linspace over the
% maturity figure log-range); the first selected state maximizes squared
% distance from the mean curve and each further state maximizes its minimum
% squared distance from the already-selected curves. Curve distances are
% unchanged by adding a constant or reversing sign, so selection on h equals
% selection on the plotted curves. Integer legend labels are
% labels only.

design = get_frozen_design();
maturity = get_empirical_maturity_config(cfg.real.horizon);
sel = design.vix_selector;

logRange = maturity.figure_log_range;
step = (logRange(2) - logRange(1)) / (sel.num_eval - 1);
mLog = (0:sel.num_eval-1) * step + logRange(1);
mLog(end) = logRange(2);
mRaw = exp(mLog(:));

vLo = min(model_state.exo_grid(:));
vHi = max(model_state.exo_grid(:));
cand = (0:sel.n_candidates-1) * ((vHi - vLo) / (sel.n_candidates - 1)) + vLo;
cand(end) = vHi;

curves = zeros(sel.n_candidates, sel.num_eval);
for c = 1:sel.n_candidates
    exo = build_exo_representer(cfg, model_state, [], ...
        struct('mode', 'conditional', 'value', cand(c)));
    out = evaluate_sdf_grid(cfg, model_state, mRaw, exo);
    curves(c, :) = out.h(:).';
end

meanCurve = mean(curves, 1);
d0 = sum((curves - meanCurve).^2, 2);
[~, first] = max(d0);
selected = first;
for s = 2:sel.n_levels
    minDists = inf(sel.n_candidates, 1);
    for q = 1:numel(selected)
        minDists = min(minDists, sum((curves - curves(selected(q), :)).^2, 2));
    end
    minDists(selected) = -1;
    [~, nxt] = max(minDists);
    selected(end+1) = nxt; %#ok<AGROW>
end

levels = sort(cand(selected));
levels = levels(:).';

assert(numel(unique(levels)) == sel.n_levels, ...
    'VIX selector returned a repeated state.');

info = struct();
info.candidates = cand;
info.selected_indices = selected;
info.levels = levels;
info.display_labels = compose('%.0f', levels);
info.exo_grid_range = [vLo, vHi];
info.figure_log_range = logRange;

end
