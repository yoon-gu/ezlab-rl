V_eff = readmatrix('../data/vaccine_efficacy.xlsx');
params.e1 = V_eff(:,2);
params.e2 = V_eff(:,3);
params.e3 = V_eff(:,4);

figure1 = figure('pos', [10 10 1200 450]);
plot(1:471,params.e1,'.-',34:471,params.e2(34:471),'.-',240:471,params.e3(240:471),'.-')
xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
axis([0 471 0 1])
xlabel('Month of 2021 and 2022')
ylabel('Prevention rate of vaccine(%)')
set(gca, 'FontSize', 11)

saveas(gca, sprintf('vac_eff.eps'), 'epsc')