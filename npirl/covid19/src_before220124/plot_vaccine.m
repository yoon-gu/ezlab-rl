clear; close all; clc;
addpath ../data
V_num = readmatrix('../data/vaccination_number.xlsx');
v1 = V_num(:,2);
v2 = V_num(:,3);
v3 = V_num(:,4);
clear V_num

figure1 = figure('pos', [10 10 1200 850]);
subplot(3,1,1)
plot(1:471,v1(1:471),'.-');
xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
axis([0 339 0 5e+5])
xlabel('Month of 2021 and 2022')
ylabel('Vaccination (1st dose)')
set(gca, 'FontSize', 11)

subplot(3,1,2)
plot(1:471,v2(1:471),'.-');
xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
axis([0 339 0 5e+5])
xlabel('Month of 2021 and 2022')
ylabel('Vaccination (2nd dose)')
set(gca, 'FontSize', 11)

subplot(3,1,3)
plot(1:471,v3(1:471),'.-');
xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
axis([0 339 0 5e+5])
xlabel('Month of 2021 and 2022')
ylabel('Vaccination (3rd dose - booster)')
set(gca, 'FontSize', 11)

saveas(gca, sprintf('vaccine.eps'), 'epsc')

