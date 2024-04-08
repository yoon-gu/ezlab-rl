clear; close all; clc;
addpath ../data ../estimate
path = '../result/result1104';

V_num = readmatrix('../data/vaccination_number.xlsx');
params.v1_num = V_num(:,2);
params.v2_num = V_num(:,3);

V_eff = readmatrix('../data/vaccine_efficacy.xlsx');
params.e1 = V_eff(:,2);
params.e2 = V_eff(:,3);

alpha_delta_eff = readmatrix('../data/alpha_delta_effect.xlsx');
params.alpha_eff = alpha_delta_eff(:,2);
params.delta_eff = alpha_delta_eff(:,3);


figure1 = figure('pos', [10 10 1000 500]);
subplot(2,1,1);
plot(1:length(params.v1_num),params.v1_num,'.-','MarkerSize',10);
xticks([0 14 45 75 106 137 168 198 229 260 290])
xticklabels({'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'})
axis([0 199 0 4e+5])
xlabel('day')
ylabel('number of vaccination')
legend('1st dose','Location','northwest')
set(gca, 'FontSize', 13)
subplot(2,1,2);
plot(1:length(params.v2_num),params.v2_num,'.-','MarkerSize',10);
xticks([0 14 45 75 106 137 168 198 229 260 290])
xticklabels({'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'})
axis([0 199 0 4e+5])
xlabel('day')
ylabel('number of vaccination')
legend('2nd dose','Location','northwest')
set(gca, 'FontSize', 13)
saveas(gca, sprintf('%s/vaccination.eps',path), 'epsc')

figure1 = figure('pos', [10 10 600 350]);
plot(1:length(params.e1),params.e1','.-','MarkerSize',8);
hold on;
plot(1:length(params.e2),params.e2','.-','MarkerSize',8);
xticks([0 14 45 75 106 137 168 198 229 260 290])
xticklabels({'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'})
axis([0 199 0 1])
xlabel('day')
ylabel('number of vaccination')
legend('1st dose','2nd dose')
set(gca, 'FontSize', 13)
saveas(gca, sprintf('%s/vaccination_efficacy.eps',path), 'epsc')

figure1 = figure('pos', [10 10 600 350]);
plot(1:length(params.delta_eff),params.delta_eff','.-','MarkerSize',8);
xticks([0 14 45 75 106 137 168 198 229 260 290])
xticklabels({'Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'})
axis([0 199 0 1])
xlabel('day')
ylabel('Proportion of delta-variant')
set(gca, 'FontSize', 13)
saveas(gca, sprintf('%s/delta_effect.eps',path), 'epsc')
