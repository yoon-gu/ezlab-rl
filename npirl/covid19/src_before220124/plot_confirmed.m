clear; close all; clc;
addpath ../data
T_seoul = readmatrix('../data/data_seoul.xlsx');
T_seoul(ismissing(T_seoul)) = 0;                    %delete "NaN"
T_seoul1 = T_seoul(357:462,2:10);
T_seoul2 = T_seoul(463:642,2:10);
T_seoul3 = T_seoul(643:694,2:10);

T_gyeonggi = readmatrix('../data/data_gyeonggi.xlsx');
T_gyeonggi(ismissing(T_gyeonggi)) = 0;       %delete "NaN"
T_gyeonggi1 = T_gyeonggi(388:493,2:10);
T_gyeonggi2 = T_gyeonggi(494:673,2:10);
T_gyeonggi3 = T_gyeonggi(674:725,2:10);

T1 = [T_seoul1;T_seoul2;T_seoul3];
T2 = [T_gyeonggi1;T_gyeonggi2;T_gyeonggi3];
T = T1+T2;
sT = sum(T,2);

figure1 = figure('pos', [10 10 1200 450]);
plot(0:337,sT,'r:.','MarkerSize',10);
xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
axis([0 338 0 6000])
xlabel('Month of 2021 and 2022')
ylabel('Number of confirmed cases')
set(gca, 'FontSize', 11)

saveas(gca, sprintf('confirmed.eps'), 'epsc')

