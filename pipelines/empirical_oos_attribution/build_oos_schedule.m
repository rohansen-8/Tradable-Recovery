function sched = build_oos_schedule(oos, panel)
%% Expanding-window schedule: one OOS date and one training set per window.
%
% Window w evaluates OOS date tau = dates(K + w), K = ceil(train_frac*T), so
% window 1 trains on the first K dates and the last window evaluates dates(T)
% (oos_run.py: range(K, T) in 0-based indexing).
% Its training set is every earlier date s with
%   'realized_payoff' : realization_date(s) <= tau (payoff observable at the
%                       formation close of tau);
%   'python_reference': no further condition (oos_run.py).
% train_key identifies the training set exactly (it enters the window
% fingerprint), so a 'python_reference' run reuses a primary fit whose
% training set is identical.

D = panel.dates_table;
T = panel.T;
K = ceil(oos.protocol.train_frac * T);
assert(K < T, 'build_oos_schedule:NoOOS', 'Initial window covers all %d dates.', T);
nW = T - K;

w = struct('id', cell(nW,1), 'oos_index', [], 'oos_date', [], 'train_index', [], ...
    'train_key', [], 'n_train', [], 'n_excluded_unrealized', [], 'min_slack_days', []);
for k = 1:nW
    i = K + k;
    tau = D.date(i);
    earlier = (1:i-1).';
    realized = D.realization_date(earlier) <= tau;
    switch oos.protocol.information_rule
        case 'realized_payoff',  idx = earlier(realized);
        case 'python_reference', idx = earlier;
    end
    assert(all(D.date(idx) < tau), 'build_oos_schedule:LookAhead', 'Training date >= OOS date.');
    if strcmp(oos.protocol.information_rule, 'realized_payoff')
        assert(all(D.realization_date(idx) <= tau), 'build_oos_schedule:Unrealized', ...
            'Training payoff not realized by the OOS formation date.');
    end
    w(k).id = k;
    w(k).oos_index = i;
    w(k).oos_date = tau;
    w(k).train_index = idx;
    w(k).train_key = strjoin(cellstr(string(D.date(idx), 'yyyyMMdd')), ',');
    w(k).n_train = numel(idx);
    w(k).n_excluded_unrealized = sum(~realized);
    w(k).min_slack_days = days(tau - max(D.realization_date(idx)));
end

sched = struct();
sched.K = K;
sched.T = T;
sched.n_windows = nW;
sched.windows = w;
sched.n_windows_embargo_binds = sum([w.n_excluded_unrealized] > 0);

end
