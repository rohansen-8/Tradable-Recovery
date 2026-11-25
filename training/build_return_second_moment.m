function moments = build_return_second_moment(alpha)
%% Build alpha_bar and the uncentered second moment Q_alpha.
%
%   alpha_bar = T^{-1} sum_t alpha_t
%   Q_alpha   = T^{-1} sum_t alpha_t alpha_t'

    if isstruct(alpha)
        alpha = alpha.alpha;
    end

    if isempty(alpha) || ~ismatrix(alpha)
        error('build_return_second_moment:BadAlpha', 'alpha must be an Nh-by-T matrix.');
    end

    T = size(alpha, 2);
    alpha_bar = mean(alpha, 2);
    Q_alpha = (alpha * alpha.') / T;
    Q_alpha = 0.5 * (Q_alpha + Q_alpha.');

    moments = struct();
    moments.T = T;
    moments.alpha_bar = alpha_bar;
    moments.Q_alpha = Q_alpha;
end
