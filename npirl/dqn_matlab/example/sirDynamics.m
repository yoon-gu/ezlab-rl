function dydt = sirDynamics(t, y, action, beta, gamma)
    S = y(1);
    I = y(2);
    R = y(3);
    V = action;  % 백신 접종률을 결정하는 에이전트의 액션
    
    dSdt = -beta * S * I - V * S;
    dIdt = beta * S * I - gamma * I;
    dRdt = gamma * I + V * S;
    
    dydt = [dSdt; dIdt; dRdt];
end
