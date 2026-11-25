function comp = evaluate_oos_payoff_components(oos, model_state, dfTau, halfSpread)
%% Realized OOS payoff of one fitted window and its exact decomposition.
%
% The realized payoff is alpha_tau' h_star, with alpha_tau built by the
% production build_alpha_panel from the OOS date's rows, using the window's
% basis grids and length scales (model_state.cfg). Because alpha_tau is linear
% in (Delta_tau, w_j * RO_j), masked copies of the same rows give the parts:
%   forward  : option payoffs set to 0            -> Delta * h_m(1, v)
%   cell b,k : Delta and payoffs outside (b,k) 0  -> sum_{j in b,k} w_j RO_j h_mm(m_j, v)
% All copies go through ONE build_alpha_panel call (distinct placeholder
% dates as group labels), so total = forward + sum of cells holds by
% linearity, up to rounding (identity 1).
%
% Strike buckets b use the fixed cutoffs c1 < c2 and test domain [a, z]:
%   b1 m < a | b2 a <= m < c1 | b3 c1 <= m < c2 | b4 c2 <= m <= z | b5 m > z
% and k splits each bucket by option type (flag 0 put, 1 call), because the
% panel's options are out of the money by spot, not by forward moneyness,
% so near the forward both types occur.
% Positions h_m(1, v) and h_mm(m_j, v) come from the production evaluator
% (evaluate_sdf_derivatives, conditional VIX, no clipping); identity 2 checks
% them against the payoff: Delta*h_m(1) + sum_j w_j RO_j h_mm(m_j) = total.

cfgF = model_state.cfg;
p = oos.protocol;
a = p.test_domain(1); z = p.test_domain(2);
c1 = p.region_cutoffs(1); c2 = p.region_cutoffs(2);
m = dfTau.frwd_moneyness;
assert(isscalar(unique(dfTau.date)), 'evaluate_oos_payoff_components:Date', 'One OOS date expected.');
assert(issorted(m), 'evaluate_oos_payoff_components:Order', 'Rows must be sorted by moneyness.');
assert(all(ismember(dfTau.flag, [0 1])), 'evaluate_oos_payoff_components:Flag', 'flag must be 0/1.');

bucket = 1 + (m >= a) + (m >= c1) + (m >= c2) + (m > z);
isCall = dfTau.flag == 1;
nB = 5;
labels = local_labels(a, c1, c2, z);

% Stack: copy 1 = full, copy 2 = forward only, copies 3.. = (bucket, type) cells.
nCopy = 2 + 2 * nB;
J = height(dfTau);
base = dfTau(:, {'date','frwd_moneyness','OPT_NET_R_PAYOFF','weight','SP500_ret','VIX'});
stack = repmat(base, nCopy, 1);
copyId = repelem((1:nCopy).', J);
stack.date = base.date(1) + days(copyId - 1);
RO = stack.OPT_NET_R_PAYOFF;
RO(copyId == 2) = 0;
stack.SP500_ret(copyId >= 3) = 0;
for b = 1:nB
    for k = 0:1
        cid = 3 + 2 * (b - 1) + k;
        inCell = repmat(bucket == b & isCall == logical(k), nCopy, 1);
        RO(copyId == cid & ~inCell) = 0;
    end
end
stack.OPT_NET_R_PAYOFF = RO;

ap = build_alpha_panel(cfgF, stack, model_state.mon_grid, model_state.exo_grid);
assert(ap.T == nCopy, 'evaluate_oos_payoff_components:Copies', 'Unexpected group count.');
pay = ap.alpha.' * model_state.h_star(:);

comp = struct();
comp.total = pay(1);
comp.forward = pay(2);
comp.cells = reshape(pay(3:end), 2, nB).';          % nB-by-2, columns put/call
comp.bucket_labels = labels;
comp.identity1_residual = comp.total - comp.forward - sum(comp.cells(:));

% Positions from the production evaluator (no VIX clipping).
v = dfTau.VIX(1);
exo = build_exo_representer(cfgF, model_state, [], struct('mode', 'conditional', 'value', v));
d = evaluate_sdf_derivatives(cfgF, model_state, [1; m], exo);
comp.fwd_position = d.h_m(1);
hmm = d.h_mm(2:end);
contrib = dfTau.weight .* dfTau.OPT_NET_R_PAYOFF .* hmm;
comp.identity2_residual = comp.total - (dfTau.SP500_ret(1) * comp.fwd_position + sum(contrib));
comp.identity3_residual = comp.forward - dfTau.SP500_ret(1) * comp.fwd_position;

% Per-option record and worst-case entry cost at the full half-spread
% (OptPortOpt_V2 tc_oos_run convention: sum_j |h_mm w_j| * half_spread_j).
comp.options = table(m, dfTau.flag, bucket, dfTau.weight, dfTau.OPT_NET_R_PAYOFF, ...
    hmm, contrib, halfSpread, 'VariableNames', ...
    {'m','flag','bucket','weight','RO','h_mm','contribution','half_spread'});
cost = abs(hmm .* dfTau.weight) .* halfSpread;
comp.cost_cells = [accumarray(bucket(~isCall), cost(~isCall), [nB 1]), ...
                   accumarray(bucket(isCall), cost(isCall), [nB 1])];
comp.n_options_cells = [accumarray(bucket(~isCall), 1, [nB 1]), accumarray(bucket(isCall), 1, [nB 1])];

end

function labels = local_labels(a, c1, c2, z)
%% Strike-range labels of the five buckets (actual ranges, not "puts"/"calls").
labels = {sprintf('m<%.4f (below test domain)', a), ...
          sprintf('%.4f<=m<%.3f', a, c1), ...
          sprintf('%.3f<=m<%.3f', c1, c2), ...
          sprintf('%.3f<=m<=%.4f', c2, z), ...
          sprintf('m>%.4f (above test domain)', z)};
end
