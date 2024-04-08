% Main file for prediction
clear all; close all; clc;
addpath ../data ../estimate

path = '../result/result0428_revision';
% mkdir(path);
path_est = sprintf('%s/est', path);

%% 1. Loading - estimation information and etc
% 1-1. estimation information
load(sprintf('%s/calibration.mat',path_est))

params.beta = beta_MLE;
params.delta = delta_MLE;

clearvars -except params

path = '../result/result0428_revision';
mkdir(path);
path_pred = sprintf('%s/pred/calibration', path);
mkdir(path_pred);

% 1-1 vaccination number
V_num = readmatrix('../data/vaccination_number.xlsx');
params.v1_num = V_num(:,2);
params.v2_num = V_num(:,3);
% params.v1_num(208:end) = params.v1(208:end)/3;
% params.v2_num(208:end) = params.v2(208:end)/2;

% 1-2. vaccination strategy and vaccine efficacy
% prediction : vaccination 유지
V_1st_ratio = readmatrix('../data/1st_dose_ratio_by_age.xlsx');
params.v1_ratio = V_1st_ratio(2:end,2:10);

V_2nd_ratio = readmatrix('../data/2nd_dose_ratio_by_age.xlsx');
params.v2_ratio = V_2nd_ratio(2:end,2:10);

params.V1 = [zeros(14,9); params.v1_num .* params.v1_ratio];
params.V2 = [zeros(14,9); params.v2_num .* params.v2_ratio];

V_eff = readmatrix('../data/vaccine_efficacy.xlsx');
params.e1 = V_eff(:,2);
params.e2 = V_eff(:,3);

% 1-3 alpha- and delta- variant proportion
alpha_delta_eff = readmatrix('../data/alpha_delta_effect.xlsx');
params.alpha_eff = alpha_delta_eff(:,2);
params.delta_eff = alpha_delta_eff(:,3);

% 1-4 etc (end of january : 351 end of april : 440, end of may : 471)
params.tspan = 0 : params.dt : 440;
params.time_stamp = 0 : 440;
params.sd = params.sd(1:259);

% 1-5 beds
% beds1 = readmatrix('../data/bed1.xlsx');
% beds2 = readmatrix('../data/bed2.xlsx');
% beds3 = readmatrix('../data/bed3.xlsx');
% 
% params.severe_bed_using = (sum(beds1(:,5:6),2) + sum(beds2(:,5:6),2));
% params.severe_bed_all = (sum(beds1(:,2:3),2) + sum(beds2(:,2:3),2));

params.sc_index = 260;

save(sprintf('%s/variables_experiment.mat',path_pred))