cas_unc = readtable("..\data\확진자\확진자정리_조사날근처.xlsx");

x = @(f,d0,d1,n0,n1) (1-f)/(1-d1*n1/(d0*n0+d1*n1)*f);

time_table = cas_unc.date;
time_at_july_survey = find(time_table==datetime("2021-07-26"));
time_at_sep_survey1 = find(time_table==datetime("2021-09-27"));
time_at_sep_survey2 = find(time_table==datetime("2021-09-28"));
time_at_oct_survey = find(time_table==datetime("2021-10-01"));


cas = cas_unc.confirmedCases;
unc = cas_unc.untraceableProportion;
unc(unc==1) = nan;
ideal_d0 = 2.3; ideal_d1 = 4.9;
mean_all_unc = mean(unc(~isnan(unc)));
ideal_x = x(mean_all_unc, ideal_d0, ideal_d1, 1, 1);

nrow = height(cas_unc);

real_x = zeros(nrow-6,1);
d0 = 2; d1 = 5;
for i=1:nrow-6
    now_cas = cas(i:(i+6));
    n0 = mean(now_cas(end-(d0-1):end));
    n1 = mean(now_cas(1:end-d0));
    now_unc = unc(i:(i+6));
    mean_unc = mean(now_unc(~isnan(now_unc)));
    real_x(i) = 100*x(mean_unc, d0, d1, n0, n1);
end
asymptomatic_prop = 100-real_x;

%%
figure('Position', [642 42 1278 953], 'Units', 'pixels');
hold on;

b = bar(cas_unc.date, table2array(cas_unc(:,5:end)), ...
    'stacked', "BarWidth", 1,'FaceAlpha', 0.3);

ytickformat("%d%%")
ylim([0,100])

yyaxis right

set(gca, "YColor", "#bc5090")

plot(time_table(7:end), asymptomatic_prop, "-", "LineWidth", 2, ...
    "Color", "#bc5090");
text(time_table(end-90), asymptomatic_prop(end-60)+10, ...
    "Asymptomatic proportion", "FontWeight", "bold", "Color", "#bc5090");

plot(time_table(time_at_july_survey), ...
    asymptomatic_prop(time_at_july_survey-6), ...
    "kx", "LineWidth", 2, "MarkerSize", 10);
plot(time_table(time_at_sep_survey1), ...
    asymptomatic_prop(time_at_sep_survey1-6), ...
    "kx", "LineWidth", 2, "MarkerSize", 10);
plot(time_table(time_at_sep_survey2), ...
    asymptomatic_prop(time_at_sep_survey2-6), ...
    "kx", "LineWidth", 2, "MarkerSize", 10);
plot(time_table(time_at_oct_survey), ...
    asymptomatic_prop(time_at_oct_survey-6), ...
    "kx", "LineWidth", 2, "MarkerSize", 10);

ytickformat("%d%%")

xtickformat("MM-W");

legend(b,{"WT", "α", "β", "γ", "δ", "ο"}, ...
    "Location", "northoutside", ...
    "Orientation", "horizontal");

set(gca,'LooseInset',get(gca,'TightInset'));
set(findall(gcf,'-property','FontSize'),'FontSize',20);

exportgraphics(gcf,"asymptomatic_proportion.png");

close gcf;

%%
figure('Position', [642 42 1278 953], 'Units', 'pixels');
hold on;

b = bar(cas_unc.date, table2array(cas_unc(:,5:end)), ...
    'stacked', "BarWidth", 1,'FaceAlpha', 0.3);

ytickformat("%d%%")
ylim([0,100])

yyaxis right

set(gca, "YColor", "#bc5090")

plot(time_table, cas_unc.confirmedCases, "-", "LineWidth", 2, ...
    "Color", "#bc5090");
text(time_table(end-110), cas_unc.confirmedCases(end-60)+900, ...
    "Daily confirmed cases", "FontWeight", "bold", "Color", "#bc5090");

xtickformat("MM-W");

legend(b,{"WT", "α", "β", "γ", "δ", "ο"}, ...
    "Location", "northoutside", ...
    "Orientation", "horizontal");

set(gca,'LooseInset',get(gca,'TightInset'));
set(findall(gcf,'-property','FontSize'),'FontSize',20);

exportgraphics(gcf,"confirmed_cases.png");

close gcf;