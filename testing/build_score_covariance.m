function [Omega_hat, S_use, info] = build_score_covariance(S_score, dates, opts)
%% Estimate the covariance matrix for shape-test score rows.
%
%   S_score is T-by-r. Each row is the finite-grid influence-score vector.
%   The default is the lag-zero covariance:
%       Omega_hat = S_score' * S_score / T.
%
%   opts.type:
%       'iid_lag0'      lag-zero covariance
%       'hac_bartlett'  Bartlett HAC long-run covariance
%
%   opts.bandwidth:
%       nonnegative integer HAC truncation lag. Ignored for iid_lag0.
%
%   opts.winsor_margin:
%       0 for no winsorization; e.g. 0.01 caps the top 1% score-row norms.
%
%   opts.recenter:
%       false by default, so iid_lag0 with zero winsorization exactly matches
%       the uncentered lag-zero covariance above. Set true to subtract the sample mean score before covariance.
%
%   opts.psd_cleanup:
%       default false for iid_lag0, true for HAC with positive bandwidth.
%       Cleanup clips negative eigenvalues of the symmetric HAC estimate to zero.

    if nargin < 2 || isempty(dates)
        dates = [];
    end
    if nargin < 3 || isempty(opts)
        opts = struct();
    end

    S_score = double(S_score);
    [T, r] = size(S_score);

    if T < 1 || r < 1
        error('build_score_covariance:BadScores', 'S_score must be a nonempty T-by-r matrix.');
    end
    if any(~isfinite(S_score(:)))
        error('build_score_covariance:BadScores', 'S_score contains nonfinite values.');
    end

    if isempty(dates)
        dates = (1:T).';
    end
    if numel(dates) ~= T
        error('build_score_covariance:BadDates', 'dates must have one entry per score row.');
    end

    cov_type = char(lower(string(get_opt(opts, 'type', 'iid_lag0'))));
    requested_bandwidth = double(get_opt(opts, 'bandwidth', 0));
    winsor_margin = double(get_opt(opts, 'winsor_margin', 0));
    recenter = logical(get_opt(opts, 'recenter', false));

    if winsor_margin < 0 || winsor_margin >= 0.5
        error('build_score_covariance:BadWinsorMargin', 'winsor_margin must be in [0, 0.5).');
    end
    if requested_bandwidth < 0 || requested_bandwidth ~= round(requested_bandwidth)
        error('build_score_covariance:BadBandwidth', 'bandwidth must be a nonnegative integer.');
    end

    S_use = S_score;
    score_norm = sqrt(sum(S_use.^2, 2));

    if winsor_margin == 0
        cap = Inf;
        scale = ones(T, 1);
        capped = false(T, 1);
    else
        cap = quantile(score_norm, 1 - winsor_margin);
        scale = min(1, cap ./ max(score_norm, eps));
        capped = score_norm > cap;
        S_use = S_use .* scale;
    end

    if recenter
        S_use = S_use - mean(S_use, 1);
    end

    switch cov_type
        case {'iid_lag0', 'lag0', 'iid'}
            cov_type = 'iid_lag0';
            bandwidth = 0;
            Omega_hat = (S_use.' * S_use) / T;

        case {'hac_bartlett', 'hac'}
            cov_type = 'hac_bartlett';
            bandwidth = min(requested_bandwidth, T - 1);
            Omega_hat = (S_use.' * S_use) / T;
            for ell = 1:bandwidth
                weight = 1 - ell / (bandwidth + 1);
                Gamma = (S_use((ell + 1):T, :).' * S_use(1:(T - ell), :)) / T;
                Omega_hat = Omega_hat + weight * (Gamma + Gamma.');
            end

        otherwise
            error('build_score_covariance:BadType', 'Unknown covariance type: %s.', cov_type);
    end

    Omega_hat = 0.5 * (Omega_hat + Omega_hat.');

    default_cleanup = strcmp(cov_type, 'hac_bartlett') && bandwidth > 0;
    psd_cleanup = logical(get_opt(opts, 'psd_cleanup', default_cleanup));

    eig_raw = eig(Omega_hat);
    min_eig_raw = min(eig_raw);
    max_eig_raw = max(eig_raw);
    psd_cleaned = false;
    psd_negative_mass = sum(abs(eig_raw(eig_raw < 0)));

    if psd_cleanup && min_eig_raw < 0
        [V, D] = eig(Omega_hat);
        d = diag(D);
        d_clean = max(d, 0);
        Omega_hat = V * diag(d_clean) * V.';
        Omega_hat = 0.5 * (Omega_hat + Omega_hat.');
        psd_cleaned = true;
    end

    eig_clean = eig(Omega_hat);

    trace_denom = sum(score_norm.^2);
    if trace_denom > 0
        trace_share = score_norm.^2 / trace_denom;
    else
        trace_share = zeros(T, 1);
    end

    score_table = table();
    score_table.row_index = (1:T).';
    score_table.date = dates(:);
    score_table.score_norm = score_norm(:);
    score_table.trace_share = trace_share(:);
    score_table.winsor_margin = repmat(winsor_margin, T, 1);
    score_table.cap = repmat(cap, T, 1);
    score_table.scale = scale(:);
    score_table.capped = capped(:);

    info = struct();
    info.type = cov_type;
    info.bandwidth = bandwidth;
    info.requested_bandwidth = requested_bandwidth;
    info.kernel = 'bartlett';
    info.T = T;
    info.n_grid = r;
    info.winsor_margin = winsor_margin;
    info.cap_probability = 1 - winsor_margin;
    info.score_norm_cap = cap;
    info.n_capped_dates = sum(capped);
    info.frac_capped_dates = mean(capped);
    info.capped_trace_share = sum(trace_share(capped));
    info.recentered = recenter;
    info.psd_cleanup = psd_cleanup;
    info.psd_cleaned = psd_cleaned;
    info.psd_negative_mass_raw = psd_negative_mass;
    info.min_eig_raw = min_eig_raw;
    info.max_eig_raw = max_eig_raw;
    info.min_eig = min(eig_clean);
    info.max_eig = max(eig_clean);
    info.score_table = score_table;
    info.capped_dates = score_table(capped, :);
end

function value = get_opt(opts, name, defaultValue)
    if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = defaultValue;
    end
end
