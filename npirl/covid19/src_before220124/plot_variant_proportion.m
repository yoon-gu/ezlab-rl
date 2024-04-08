t1 = 0 : 6;
d = [0.2 1.1 1.7 1.8 4 12.5 26.7];
t2 = 0 : 0.01 : 8;
fun = @(r)r*(exp(t1)-1)+0.2-d;
x0 = 0.01;
x = lsqnonlin(fun,x0);

figure1 = figure('pos', [10 10 600 350]);
plot(t1,d,'.',t2,x*(exp(t2)-1)+0.2,'MarkerSize',8);
xticks([0 1 2 3 4 5 6 7 8])
xticklabels({'Dec/1', 'Dec/2', 'Dec/3', 'Dec/4', 'Dec/5', 'Jan/1', 'Jan/2', 'Jan/3', 'Jan/4'})
xlabel('month/week')
ylabel('Proportion(%)')
legend('Proportion of omicron','fitting')
set(gca, 'FontSize', 12)
saveas(gca, sprintf('variant_proportion.eps'), 'epsc')