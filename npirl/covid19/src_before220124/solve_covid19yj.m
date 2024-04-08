function result = solve_covid19yj(theta,M,index)
% for 2/15-5/31 or 6/1-9/1
switch index
    case 1
        beta = theta;
        delta = M.delta;
        M.time_stamp = 0 : M.length1;
    case 2
        delta = theta;
        beta = M.beta;
end
% Initialize states
S = zeros(M.num_grp, length(M.time_stamp), 1/M.dt+1);
E = zeros(size(S));
I = zeros(size(S));
H = zeros(size(S));
R = zeros(size(S));
EV1 = zeros(size(S));
IV1 = zeros(size(S));
EV2 = zeros(size(S));
IV2 = zeros(size(S));
V1 = zeros(size(S));
V2 = zeros(size(S));
new_inf = zeros(size(S));
F = zeros(size(S));
SI = zeros(size(S));

% Initial states
S(:,1,1) = M.S0;
E(:,1,1) = M.E0;
I(:,1,1) = M.I0;
H(:,1,1) = M.H0;
R(:,1,1) = M.R0;
EV1(:,1,1) = M.EV10;
IV1(:,1,1) = M.IV10;
EV2(:,1,1) = M.EV20;
IV2(:,1,1) = M.IV20;
V1(:,1,1) = M.V10;
V2(:,1,1) = M.V20;
new_inf(:,1,1) = M.new_inf0;
F(:,1,1) = M.F0;
SI(:,1,1) = M.SI0;

% History of negative flag
neg_flag_S_hist = false( M.num_grp, 1);
neg_flag_V1_hist = false( M.num_grp, 1);
neg_flag_S = false(M.num_grp, 1);
neg_flag_V1 = false(M.num_grp, 1);

i = 0;
% Loop for time_stamp
while i <= length(M.time_stamp)-2
    i = i + 1;
    % WAIFW
    if i == M.sc_index
        M.contact(2,2) = M.sc_rate * M.contact(2,2);
    end
    WAIFW = (M.alpha_eff(i) + delta * M.delta_eff(i)) * beta * M.sd(i) * M.contact;
    % Loop for tspan
    for j = 1 : length(1:1/M.dt)
        lambdaS = WAIFW * (I(:,i,j) + IV1(:,i,j) + IV2(:,i,j)) .* S(:,i,j);
        lambdaV1 = WAIFW * (1 - M.e1(i)) * (I(:,i,j) + IV1(:,i,j) + IV2(:,i,j)) .* V1(:,i,j);
        lambdaV2 = WAIFW * (1 - M.e2(i)) * (I(:,i,j) + IV1(:,i,j) + IV2(:,i,j)) .* V2(:,i,j);
        S(:,i,j+1) = S(:,i,j) + M.dt * (-lambdaS - M.V1(i,:)');
        E(:,i,j+1) = E(:,i,j) + M.dt * (lambdaS - M.kappa* E(:,i,j));
        I(:,i,j+1) = I(:,i,j) + M.dt * (M.kappa* E(:,i,j) - M.alpha * I(:,i,j));
        H(:,i,j+1) = H(:,i,j) + M.dt * (M.alpha * (I(:,i,j) + IV1(:,i,j) + IV2(:,i,j)) - M.gamma * H(:,i,j));
        R(:,i,j+1) = R(:,i,j) + M.dt * (M.gamma * H(:,i,j));
        V1(:,i,j+1) = V1(:,i,j) + M.dt * (M.V1(i,:)' - lambdaV1 - M.V2(i,:)');
        V2(:,i,j+1) = V2(:,i,j) + M.dt * (M.V2(i,:)' - lambdaV2);
        EV1(:,i,j+1) = EV1(:,i,j) + M.dt * (lambdaV1 - M.kappa * EV1(:,i,j));
        EV2(:,i,j+1) = EV2(:,i,j) + M.dt * (lambdaV2 - M.kappa * EV2(:,i,j));
        IV1(:,i,j+1) = IV1(:,i,j) + M.dt * (M.kappa * EV1(:,i,j) - M.alpha * IV1(:,i,j));
        IV2(:,i,j+1) = IV2(:,i,j) + M.dt * (M.kappa * EV2(:,i,j) - M.alpha * IV2(:,i,j));
%         new_inf(:,i,j+1) = new_inf(:,i,j) + M.dt * (M.alpha * (I(:,i,j) + IV1(:,i,j) + IV2(:,i,j)));
        new_inf(:,i,j+1) = new_inf(:,i,j) + M.dt * (M.alpha * (I(:,i,j) + IV1(:,i,j) + IV2(:,i,j)) + M.alpha * (I(:,i,j+1) + IV1(:,i,j+1) + IV2(:,i,j+1))) / 2;
        F(:,i,j+1) = F(:,i,j) + M.dt * (M.alpha * (I(:,i,j) + IV1(:,i,j) + IV2(:,i,j)) .* M.fatality_rate');
        SI(:,i,j+1) = SI(:,i,j) + M.dt * (M.alpha * (I(:,i,j) + (1-M.v_prevent(1)) * IV1(:,i,j) + (1-M.v_prevent(2)) * IV2(:,i,j)) .* M.severe_illness_rate');
       
        % flag for negative compartment
        if (any(S(:,i,j+1) < 0) || any(V1(:,i,j+1) < 0))
            neg_flag_S = (S(:,i,j+1) < 0);
            neg_flag_V1 = (V1(:,i,j+1) < 0);
            break;
        end
    end
    
    if (any(neg_flag_S) || any(neg_flag_V1))
        % update vaccineation number
        [M.V1, M.V2, neg_flag_S_hist, neg_flag_V1_hist] = update_vaccine(M.V1, M.V2, i, neg_flag_S, ...
            neg_flag_V1, neg_flag_S_hist, neg_flag_V1_hist);
        % Reset flag
        neg_flag_S(:) = false;
        neg_flag_V1(:) = false;
        % Reset index
        i = i - 1;
    end
    % update for next i
    S(:,i+1,1) = S(:,i,end);
    E(:,i+1,1) = E(:,i,end);
    I(:,i+1,1) = I(:,i,end);
    H(:,i+1,1) = H(:,i,end);
    R(:,i+1,1) = R(:,i,end);
    V1(:,i+1,1) = V1(:,i,end);
    V2(:,i+1,1) = V2(:,i,end);
    EV1(:,i+1,1) = EV1(:,i,end);
    EV2(:,i+1,1) = EV2(:,i,end);
    IV1(:,i+1,1) = IV1(:,i,end);
    IV2(:,i+1,1) = IV2(:,i,end);
    new_inf(:,i+1,1) = M.new_inf0;
    F(:,i+1,1) = M.new_inf0;
    SI(:,i+1,1) = M.new_inf0;
end

result.S = S(:,:,1);
result.E = E(:,:,1);
result.I = I(:,:,1);
result.H = H(:,:,1);
result.R = R(:,:,1);
result.V1 = V1(:,:,1);
result.V2 = V2(:,:,1);
result.EV1 = EV1(:,:,1);
result.EV2 = EV2(:,:,1);
result.IV1 = IV1(:,:,1);
result.IV2 = IV2(:,:,1);
result.new_inf = new_inf(:,1:end-1,end);
result.F = F(:,1:end-1,end);
result.SI = SI(:,1:end-1,end);
result.v1 = M.V1;
result.v2 = M.V2;