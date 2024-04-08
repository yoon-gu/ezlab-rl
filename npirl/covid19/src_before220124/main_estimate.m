% Main file for estimation
clear; close all; clc;
addpath ../data ../estimate

path = '../result/result0428_revision';
mkdir(path);
path_est = sprintf('%s/est', path);
mkdir(path_est);

%% 1. Loading - data and necessary informations

% 1-1 incidence data
T_seoul = readmatrix('../data/data_seoul.xlsx');
T_seoul(ismissing(T_seoul)) = 0;            % delete "NaN"
T_seoul1 = T_seoul(357:462,2:10);
T_seoul2 = T_seoul(463:615,2:10);
% T_seoul3 = T_seoul(616:694,2:10);

T_gyeonggi = readmatrix('../data/data_gyeonggi.xlsx');
T_gyeonggi(ismissing(T_gyeonggi)) = 0;      % delete "NaN"
T_gyeonggi1 = T_gyeonggi(388:493,2:10);
T_gyeonggi2 = T_gyeonggi(494:646,2:10);
% T_gyeonggi3 = T_gyeonggi(647:725,2:10);

params.inc_data1 = T_seoul1 + T_gyeonggi1;
params.inc_data2 = T_seoul2 + T_gyeonggi2;
% params.inc_data3 = T_seoul3 + T_gyeonggi3;

% 1-2 vaccination data(for estimation) and vaccination strategy(for control)
V_1st_ratio = readmatrix('../data/1st_dose_ratio_by_age.xlsx');
params.v1_ratio = V_1st_ratio(2:end,2:10);

V_2nd_ratio = readmatrix('../data/2nd_dose_ratio_by_age.xlsx');
params.v2_ratio = V_2nd_ratio(2:end,2:10);

V_num = readmatrix('../data/vaccination_number.xlsx');
params.v1_num = V_num(:,2);
params.v2_num = V_num(:,3);

params.V1 = [zeros(14,9); params.v1_num .* params.v1_ratio];
params.V2 = [zeros(14,9); params.v2_num .* params.v2_ratio];

% 1-3 alpha- and delta- variant proportion
alpha_delta_eff = readmatrix('../data/alpha_delta_effect.xlsx');
params.alpha_eff = alpha_delta_eff(:,2);
params.delta_eff = alpha_delta_eff(:,3);

% 1-4 vaccine efficacy
V_eff = readmatrix('../data/vaccine_efficacy.xlsx');
params.e1 = V_eff(:,2);
params.e2 = V_eff(:,3);

% 1-5 contact matrix - 2020 seoul-gyeonggi
params.contact = readmatrix('../data/contact_matrix.csv');

%% 2. Input - necessary informations

% 2-1 history of social distancing : 과거 내용 
% 1.4 --> 완화, 0.68 --> 강화
% 선행연구
% params.sd = [ones(1,136), 1.4*ones(1,11), 0.68*1.4*ones(1,112)];
% params.sd = [ones(1,136), 1.4*ones(1,11), 0.68*0.68*1.4*ones(1,112)];
params.sd = [ones(1,136), 1.4*ones(1,11), 0.68*0.68*0.68*1.4*ones(1,112)];

% 2-2 school attendence 
% 시간 time index
params.sc_index = 200;
params.sc_rate = 1;

% 2-2 case fatality rate by age
params.fatality_rate = [0 0 0 0 0.001 0.002 0.008 0.044 0.142];

% 2-3 proportion of severe illness by age
params.severe_illness_rate = [0.0001 0.0002 0.0015 0.0053 0.0119 0.0253 0.0488 0.1145 0.1491];

% 2-4 prevention rate of severe illness by vaccination
params.v_prevent_s = [0.75 0.94];

% 2-4 prevention rate of fatality rate by vaccination
params.v_prevent_f = [0.85 0.961];

% 2-6 2021.02.15 initial state and value of beta
load('../data/value_0215.mat')
temp = value_0215{2};
for i = 1 : 9
    temps(:,i) = temp(5*(i-1)+1:5*i)';
end

params.S0 = temps(1,:)';
params.E0 = temps(2,:)';
params.I0 = temps(3,:)';
params.H0 = temps(4,:)';
params.R0 = temps(5,:)';
params.V10 = zeros(1,9)';
params.V20 = zeros(1,9)';
params.EV10 = zeros(1,9)';
params.IV10 = zeros(1,9)';
params.EV20 = zeros(1,9)';
params.IV20 = zeros(1,9)';
params.new_inf0 = zeros(1,9)';
params.F0 = zeros(1,9)';
params.SI0 = zeros(1,9)';
temp = value_0215{1};
params.beta = temp(end);

%% 3. Construct a structured variable 

params.dt = 0.001;
params.tspan = 0 : params.dt : 259;
params.time_stamp = 0 : 259;
params.kappa = 1/4;
params.alpha = 1/4;
params.gamma = 1/14;
params.delta = 1;
params.num_grp = 9;
[params.length1,~] = size(params.inc_data1);
[params.length2,~] = size(params.inc_data2);
params.length = params.length1 + params.length2;
params.sc_rate = 1;

%% 4. MLE estimation

% 1. beta estimation
options = [];
% lb = 0;
% ub = 1;
% tic;
% fprintf('start MLE procedure \n')
% [beta_MLE,fval1,exitflag1,output1] = fminsearchcon(@(theta) cost_mle(theta,params,1),params.beta,lb,ub,[],[],[],options);
% t_mle = toc;
% fprintf('Computing time: %f seconds\n', t_mle)
% result1 = solve_covid19(beta_MLE,params,1);
load('beta_MLE','beta_MLE');

% 2. delta estimation
params.beta = beta_MLE;
lb = 0;
ub = 100;
tic;
fprintf('start MLE procedure \n')
[delta_MLE,fval2,exitflag2,output2] = fminsearchcon(@(theta) cost_mle(theta,params,2),3.5,lb,ub,[],[],[],options);
t_mle = toc;
fprintf('Computing time: %f seconds\n', t_mle)
result2 = solve_covid19(delta_MLE,params,2);
save('delta_MLE','delta_MLE');
save(sprintf('%s/calibration.mat',path_est))
%% beds
% 병상수 check
% severe illness beds
% beds1 = readmatrix('../data/bed1.xlsx');
% beds2 = readmatrix('../data/bed2.xlsx');
% beds3 = readmatrix('../data/bed3.xlsx');
% params.severe_bed = (sum(beds1(:,5:6),2) + sum(beds2(:,5:6),2)) ./ (sum(beds1(:,2:3),2) + sum(beds2(:,2:3),2));
% 
% fprintf('start ols procedure \n')
% temp_SI = sum(result2.SI)';
% f = @(theta) temp_SI(202:240)*theta - params.severe_bed;
% theta0 = 0.1;
% lb = 0;
% ub = 1;
% options = optimoptions('lsqnonlin');
% [theta_OLS, ~] = lsqnonlin(f,theta0,lb,ub,options);
% result2.bed_scale = theta_OLS;

%% 4. Visualization
% load(sprintf('%s/variables_experiment1.mat',path_est))
load('beta_MLE','beta_MLE');
load('delta_MLE','delta_MLE');

result2 = solve_covid19(delta_MLE,params,2);
% 1. fitting
% module_plot('fitting',params,result1,{'1', '1'},path_est)
% module_plot('fitting',params,result2,{'1', '2'},path_est)
module_plot('separate',params,result2,{'1', '2'},path_est)

% 2. reproduction number
Rn = reproduction_number(params,result2,beta_MLE,delta_MLE);
% module_plot('reproduction number',params,Rn,{'1', '1'},path_est)
close
close
close
close
close




   