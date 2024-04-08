%%
clc; clear; close all;
path = '../result/result0428_revision';

path_pred = sprintf('%s/pred/calibration', path);
load(sprintf('%s/variables_experiment.mat',path_pred)); 


%% sc : school, upward3 : 일상으로 돌아갔을 때
index_sc = {'sc_maintain', 'sc_upward1', 'sc_upward2', 'sc_upward3'};
% index_sc = {'sc_maintain'};

%% 
path_pred = sprintf('%s/pred/stepbystep', path);
mkdir(path_pred);
for j = 1 : length(index_sc)
    module_prediction_paper(params,index_sc{j},path_pred)
end