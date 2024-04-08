function R = reproduction_number(params,result,beta,delta)

temp1 = zeros(9);
temp2 = params.kappa * eye(9);
temp3 = params.alpha * eye(9);
temp4 = params.gamma * eye(9);

for i = 1 : length(result.S(1,:))-1
    if i == 260
        params.contact(2,2) = params.sc_rate * params.contact(2,2);
    end
    WAIFW = (params.alpha_eff(i) + delta * params.delta_eff(i)) * params.sd(i) * params.contact;
    for j = 1 : 9
        temp5(j,:) = WAIFW(j,:) * (beta * result.S(j,i) + (1-params.e1(i))*  beta * result.V1(j,i)...
            + (1-params.e2(i))*  beta * result.V2(j,i));
    end
    
    F = [temp1, temp5, temp1; temp1, temp1, temp1; temp1, temp1, temp1];
    V = [temp2, temp1, temp1; -temp2, temp3, temp1; temp1, -temp3, temp4];
    R(i) = max(abs(eig(F/V)));
end

