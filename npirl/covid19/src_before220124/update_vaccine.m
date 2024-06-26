function [vac_1st, vac_2nd, neg_flag_S_hist, neg_flag_V1_hist] = update_vaccine(vac_1st, vac_2nd, ind, neg_flag_S, neg_flag_V1, neg_flag_S_hist, neg_flag_V1_hist)
%% Update 1st vaccination number
if any(neg_flag_S)
    % Find the vaccination number for age group having negative states
    % -flag인 연령대의 vaccine의 총합
    vac_num_neg_state = sum(vac_1st(ind:end, neg_flag_S), 2);

    % Compute ratio between other age groups
    % history랑 현재랑 비교해서 neg_flag_hist를 업데이트
    neg_flag_S_hist = neg_flag_S_hist | neg_flag_S;
    % ~neg_flag : negtive flag아닌 연령대의  : 1차 vaccine / 전체 vaccine양
    ratio = vac_1st(ind:end, ~neg_flag_S_hist)./sum(vac_1st(ind:end, ~neg_flag_S_hist), 2);

    % Compute vaccination number using the ratio
    % -flag인 연령대의 vaccine의 총합 * -flag가 아닌 연령대의 ratio
    vac = vac_num_neg_state .* ratio;
    
    % Deal with zeros
    vac(isnan(vac)) = 0;

    % Update vaccination number for 1st dose
    % -flag가 아닌 연령대 vaccine = 원래 vaccine양 + 추가
    vac_1st(ind:end, ~neg_flag_S_hist) = vac_1st(ind:end, ~neg_flag_S_hist) + vac;
    % -flag인 연령대의 vaccine = 0
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
