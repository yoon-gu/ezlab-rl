% Define the step function
function [NextObs, Reward, IsDone, NextState] = sir_step(Action, State)
    % Extract the state
    S = State(1);
    I = State(2);
    R = State(3);
    
    % Parameters
    beta = 0.002;
    gamma = 0.5;
    dt = 1;
    
    % Vaccination rate from action
    v = Action;
    
    if t > 10
        v = 0.1*v;
    end
    
    % Solve SIR
    [~, y] = ode45(@(t, y) sirDynamics(t, y, v, beta, gamma), [0 dt], [S; I; R]);

    % Update the state
    S = y(end, 1);
    I = y(end, 2);
    R = y(end, 3);
    
    NextObs = [S; I; R];

    % Reward is negative of the number of infected people
    Reward = -I - v*S;

    % Episode termination condition
    IsDone = false;
    
    % Update logged signals
    NextState = NextObs;
end