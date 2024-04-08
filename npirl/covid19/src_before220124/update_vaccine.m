function [vac_1st, vac_2nd, neg_flag_S_hist, neg_flag_V1_hist] = update_vaccine(vac_1st, vac_2nd, ind, neg_flag_S, neg_flag_V1, neg_flag_S_hist, neg_flag_V1_hist)
%% Update 1st vaccination number
if any(neg_flag_S)
    % Find the vaccination number for age group having negative states
    vac_num_neg_state = sum(vac_1st(ind:end, neg_flag_S), 2);

    % Compute ratio between other age groups
    neg_flag_S_hist = neg_flag_S_hist | neg_flag_S;
    ratio = vac_1st(ind:end, ~neg_flag_S_hist)./sum(vac_1st(ind:end, ~neg_flag_S_hist), 2);

    % Compute vaccination number using the ratio
    vac = vac_num_neg_state .* ratio;
    
    % Deal with zeros
    vac(isnan(vac)) = 0;

    % Update vaccination number for 1st dose
    vac_1st(ind:end, ~neg_flag_S_hist) = vac_1st(ind:end, ~neg_flag_S_hist) + vac;
    vac_1st(ind:end, neg_flag_S_hist) = 0;
end

%% Update 2nd vaccination number
if any(neg_flag_V1)
    % Find the vaccination number for age group having negative states
    vac_num_neg_state = sum(vac_2nd(ind:end, neg_flag_V1), 2);

    % Compute ratio between other age groups
    neg_flag_V1_hist = neg_flag_V1_hist | neg_flag_V1;
    ratio = vac_2nd(ind:end, ~neg_flag_V1_hist)./sum(vac_2nd(ind:end, ~neg_flag_V1_hist), 2);

    % Compute vaccination number using the ratio
    vac = vac_num_neg_state .* ratio;

    % Deal with zeros
    vac(isnan(vac)) = 0;

    % Update vaccination number for 1st dose
    vac_2nd(ind:end, ~neg_flag_V1_hist) = vac_2nd(ind:end, ~neg_flag_V1_hist) + vac;
    vac_2nd(ind:end, neg_flag_V1_hist) = 0;
end
