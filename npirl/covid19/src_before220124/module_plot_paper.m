function module_plot_paper(type,params,result,index,path)

switch type
    case 'prediction'
        % 3-1 incidence
        data = [params.inc_data1', params.inc_data2'];
        pred = result.new_inf;
        sdata = sum(data);
        spred = sum(pred);
        for i = 1 : length(sdata)
            cdata(i) = sum(sdata(1:i));
        end
        for i = 1 : length(params.time_stamp)-1
            cpred(i) = sum(spred(1:i));
        end
 
        % 3-2 fatality and severe illness        
        sF = sum(result.F);
        sSI = sum(result.SI);
        
        for i = 1 : length(params.time_stamp)-1
            cF(i) = sum(sF(1:i));
            cSI(i) = sum(sSI(1:i));
        end
               
        figure1 = figure('pos', [10 10 1200 330]);        
        subplot(1,2,1)
        plot(1:length(params.time_stamp)-1,spred,'LineWidth',2)
        hold on;
        plot(1:length(sdata),sdata,':.','MarkerSize',10);
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Confirmed cases')
        legend('Model prediction','Fitted data','Location','northwest')
        set(gca, 'FontSize', 13)
        
        subplot(1,2,2)
        plot(1:length(params.time_stamp)-1,sSI,'LineWidth',2);
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Severe cases')
        legend('Model prediction','Location','northwest')
        set(gca, 'FontSize', 13)
        saveas(gca, sprintf('%s/%s_1.eps',path,index{1}), 'epsc')       
       
        
        figure1 = figure('pos', [10 10 1200 330]);
        subplot(1,2,1)
        plot(1:length(params.time_stamp)-1,cpred,'LineWidth',2);
        hold on;
        plot(1:length(sdata),cdata,':.','MarkerSize',10);
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Cumulative confirmed cases')
        legend('Model prediction','Fitted data','Location','northwest')
        set(gca, 'FontSize', 13)
        
        subplot(1,2,2)
        plot(1:length(params.time_stamp)-1,cF,'LineWidth',2);
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Cumulative deaths')
        legend('Model prediction','Location','northwest')
        set(gca, 'FontSize', 13)
        saveas(gca, sprintf('%s/%s_2.eps',path,index{1}), 'epsc')

        case 'reproduction number'
        Rn = result;
        figure1 = figure('pos', [10 10 800 400]);
        plot(1:length(Rn),Rn,'-','LineWidth',2);
        hold on;
        plot(0:length(Rn)-1,ones(1,length(Rn)),'LineWidth',1)
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Reproduction number')
        legend('reproduction number','Location','northwest')
        set(gca, 'FontSize', 13)
        saveas(gca, sprintf('%s/Rn_%s.eps',path,index{1}), 'epsc')

    case 'prediction_reproduction'

        data = [params.inc_data1', params.inc_data2'];
        pred = result.new_inf;
        sdata = sum(data);
        spred = sum(pred);
        for i = 1 : length(sdata)
            cdata(i) = sum(sdata(1:i));
        end
        for i = 1 : length(params.time_stamp)-1
            cpred(i) = sum(spred(1:i));
        end
 
        % 3-2 fatality and severe illness        
        sF = sum(result.F);
        sSI = sum(result.SI);
        
        for i = 1 : length(params.time_stamp)-1
            cF(i) = sum(sF(1:i));
            cSI(i) = sum(sSI(1:i));
        end
               
        load('temp_Rn');

        figure1 = figure('pos', [10 10 1800 300]);        
        subplot(1,3,1)
        plot(1:length(params.time_stamp)-1,spred,'LineWidth',2)
        hold on;
        plot(1:length(sdata),sdata,':.','MarkerSize',10);
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Confirmed cases')
        legend('Model prediction','Fitted data','Location','northwest')
        set(gca, 'FontSize', 13)
        
        subplot(1,3,2)
        plot(1:length(params.time_stamp)-1,sSI,'LineWidth',2);
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Severe cases')
        legend('Model prediction','Location','northwest')
        set(gca, 'FontSize', 13)
        saveas(gca, sprintf('%s/%s_1.eps',path,index{1}), 'epsc')       
        
        subplot(1,3,3)
        plot(1:length(Rn),Rn,'-','LineWidth',2);
        hold on;
        plot(0:length(Rn)-1,ones(1,length(Rn)),'LineWidth',1)
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Reproduction number')
        legend('reproduction number','Location','southwest')
        set(gca, 'FontSize', 13)
        saveas(gca, sprintf('%s/%s_1.eps',path,index{1}), 'epsc') 
        
        figure1 = figure('pos', [10 10 1200 330]);
        subplot(1,2,1)
        plot(1:length(params.time_stamp)-1,cpred,'LineWidth',2);
        hold on;
        plot(1:length(sdata),cdata,':.','MarkerSize',10);
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Cumulative confirmed cases')
        legend('Model prediction','Fitted data','Location','northwest')
        set(gca, 'FontSize', 13)
        
        subplot(1,2,2)
        plot(1:length(params.time_stamp)-1,cF,'LineWidth',2);
        xticks([14 45 75 106 137 168 198 229 260 290 321 352 380 411 441])
        xticklabels({'Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar','Apr','May'})
        axis([0 440 0 inf])
        xlabel('Month of 2021 and 2022')
        ylabel('Cumulative deaths')
        legend('Model prediction','Location','northwest')
        set(gca, 'FontSize', 13)
        saveas(gca, sprintf('%s/%s_2.eps',path,index{1}), 'epsc')
end
