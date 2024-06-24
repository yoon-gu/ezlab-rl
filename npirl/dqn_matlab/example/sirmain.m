clear all; clc;
close all;
%% Environment 생성
obsInfo = rlNumericSpec([3 1]);  % 상태는 S, I, R
obsInfo.Name = 'SIR States';
obsInfo.Description = 'S, I, R';
obsInfo.LowerLimit = 0;

actInfo = rlFiniteSetSpec([0 1]);  % 가능한 백신 접종률
actInfo.Name = 'Vaccination Rate';

env = rlFunctionEnv(obsInfo, actInfo, 'sir_step', 'sir_reset');

%% 난수 시드값 고정
rng(0)
Tf = 30; % episode final time

%% network 생성
net = [
    featureInputLayer(obsInfo.Dimension(1))
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(20)
    reluLayer
    fullyConnectedLayer(length(actInfo.Elements))
    ];

% dlnetwork로 변환하고 가중치 개수 표시
net = dlnetwork(net);
summary(net)

% 신경망 구성 확인
plot(net)

% 크리틱 근사기
critic = rlVectorQValueFunction(net,obsInfo,actInfo);
% 임의의 관측값 입력값을 사용하여 크리틱 확인
getValue(critic,{rand(obsInfo.Dimension)})


%% DQN 에이전트 
agent = rlDQNAgent(critic);
% 임의의 관측값 입력값을 사용하여 에이전트 확인
getAction(agent,{rand(obsInfo.Dimension)})

% DQN 에이전트 옵션 지정
agent.AgentOptions.UseDoubleDQN = false;
agent.AgentOptions.TargetSmoothFactor = 1;
agent.AgentOptions.TargetUpdateFrequency = 4;
agent.AgentOptions.ExperienceBufferLength = 1e5;
agent.AgentOptions.MiniBatchSize = 64;
agent.AgentOptions.CriticOptimizerOptions.LearnRate = 1e-3;
agent.AgentOptions.CriticOptimizerOptions.GradientThreshold = 1;

%% 에이전트 훈련 option
trainOpts = rlTrainingOptions(...
    MaxEpisodes=500, ...
    MaxStepsPerEpisode=Tf, ...
    ScoreAveragingWindowLength=5,...
    Verbose=false, ...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=0); 

%% training
trainingStats = train(agent,env,trainOpts);

%% Trainig 결과 simulation
simOptions = rlSimulationOptions(MaxSteps=Tf);
experience = sim(env,agent,simOptions);

%% Total reward
totalReward = sum(experience.Reward);

%% Plot
SIR_rl = round(experience.Observation.SIRStates.Data(:,1,:));
SIR_rl = reshape(SIR_rl,obsInfo.Dimension(1),[]);
S_rl = SIR_rl(1,:);
I_rl = SIR_rl(2,:);
R_rl = SIR_rl(3,:);
actions = experience.Action.VaccinationRate.Data(:,1,:);
actions = reshape(actions,1,[]);

%

figure(1)
plot(S_rl,'.-','LineWidth',3,'MarkerSize',20);
hold on
plot(I_rl,'.-','LineWidth',3,'MarkerSize',20)
hold off
legend("S","I")
grid on; grid minor
xlim([0 30]);
title('Dynamics RL:',num2str(totalReward, '%.2f'))
set(gca, 'FontSize', 15);

figure(2)
plot(actions, '.-','LineWidth',3,'MarkerSize',20)
grid on; grid minor
xlim([0 30]);
title('Actions')
set(gca, 'FontSize', 15);
