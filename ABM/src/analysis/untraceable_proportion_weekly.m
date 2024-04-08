cas_unc = readtable("..\data\확진자\확진자정리_조사날근처_주별.xlsx");
cas_unc_day = readtable("..\data\확진자\확진자정리_조사날근처.xlsx");

x = @(f,d0,d1,n0,n1) (1-f)/(1-d1*n1/(d0*n0+d1*n1)*f);

time_table = cas_unc.date;
time_at_july_survey = find(time_table-datetime("2021-07-26")>0, 1);
time_at_sep_survey1 = find(time_table-datetime("2021-09-27")>0, 1);
time_at_sep_survey2 = find(time_table-datetime("2021-09-28")>0, 1);
time_at_oct_survey = find(time_table-datetime("2021-10-01")>0, 1);


cas = cas_unc.confirmedCases;
unc = cas_unc.untraceableProportion;
unc(unc==1) = nan;
ideal_d0 = 2.3; ideal_d1 = 4.9;
mean_all_unc = mean(unc(~isnan(unc)));
ideal_x = x(mean_all_unc, ideal_d0, ideal_d1, 1, 1);

nrow = height(cas_unc);

real_x = zeros(nrow-6,1);
d0 = 2; d1 = 5;
for i=1:nrow
    now_time_index = find(cas_unc_day.date==cas_unc.date(1));
    now_cas = cas_unc_day.confirmedCases(now_time_index:(now_time_index+6));
    n0 = mean(now_cas(end-(d0-1):end));
    n1 = mean(now_cas(1:end-d0));
    now_unc = unc(i);
    mean_unc = mean(now_unc(~isnan(now_unc)));
    real_x(i) = 100*x(mean_unc, d0, d1, n0, n1);
end
asymptomatic_prop = 100-real_x;

%%
figure('Position', [642 42 1278 953], 'Units', 'pixels');
hold on;

b=bar(cas_unc.date, table2array(cas_unc(:,4:end)), ...
    'stacked', "BarWidth", 1,'FaceAlpha', 0.3, 'EdgeAlpha',0);

ytickformat("%d%%")

yyaxis right

set(gca, "YColor", "#bc5090")

plot(time_table, asymptomatic_prop, "-", "LineWidth", 2, ...
    "Color", "#bc5090");
text(time_table(end-10), asymptomatic_prop(end-10)-2, ...
    "Asymptomatic proportion", "FontWeight", "bold", "Color", "#bc5090");

plot(time_table(time_at_july_survey), ...
    asymptomatic_prop(time_at_july_survey), ...
    "kx", "LineWidth", 2, "MarkerSize", 10);
plot(time_table(time_at_sep_survey1), ...
    asymptomatic_prop(time_at_sep_survey1), ...
    "kx", "LineWidth", 2, "MarkerSize", 10);
plot(time_table(time_at_sep_survey2), ...
    asymptomatic_prop(time_at_sep_survey2), ...
    "kx", "LineWidth", 2, "MarkerSize", 10);
plot(time_table(time_at_oct_survey), ...
    asymptomatic_prop(time_at_oct_survey), ...
    "kx", "LineWidth", 2, "MarkerSize", 10);

ytickformat("%d%%");

xlim([time_table(1)-3.7,time_table(end)+3.7]);
xticks(time_table(4:4:end));
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

b = bar(cas_unc.date, table2array(cas_unc(:,4:end)), ...
    'stacked', "BarWidth", 1,'FaceAlpha', 0.3, 'EdgeAlpha',0);

ytickformat("%d%%")

yyaxis right

set(gca, "YColor", "#bc5090")

plot(time_table, cas_unc.confirmedCases, "-", "LineWidth", 2, ...
    "Color", "#bc5090");
text(time_table(end-18), cas_unc.confirmedCases(end-10)+2000, ...
    "Daily confirmed cases", "FontWeight", "bold", "Color", "#bc5090");

xlim([time_table(1)-3.7,time_table(end)+3.7]);
xticks(time_table(4:4:end));
xtickformat("MM-W");

legend(b,{"WT", "α", "β", "γ", "δ", "ο"}, ...
    "Location", "northoutside", ...
    "Orientation", "horizontal");

set(gca,'LooseInset',get(gca,'TightInset'));
set(findall(gcf,'-property','FontSize'),'FontSize',20);

exportgraphics(gcf,"confirmed_cases.png");

close gcf;