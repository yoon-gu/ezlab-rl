function cost = cost_mle(theta,params,index)

result = solve_covid19(theta,params,index);

switch index
    case 1
        data = params.inc_data1;
        prediction = result.new_inf';
    case 2
        data = params.inc_data2;
        prediction = result.new_inf(:,107:end)';
end


prob = poisspdf(data,prediction);
prob(prob == 0) = realmin;
cost = - sum(log(prob), 'all');