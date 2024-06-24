function [InitialObservation, InitialState] = sir_reset()
    S0 = 990;
    I0 = 10;
    R0 = 0;
    InitialState = [S0; I0; R0];
    InitialObservation = InitialState;
end