function Displaynn_multiclass(M,res,nn_params,layer_size,x,t,act_option)
tp=t*0;
layer_num=numel(layer_size); %layer number include the input and output

 % Reshape nn_params back into the matrix parameters W for each layer
para_loc_start=0;para_loc_end=0;
for iter=1:layer_num-1
    if iter ==1
        para_loc_start=1;
    else
        para_loc_start=para_loc_start+layer_size(iter)*(layer_size(iter-1)+1);
    end
    para_loc_end=para_loc_end+layer_size(iter+1)*(layer_size(iter)+1);
    
    ww{iter}=reshape(nn_params(para_loc_start:para_loc_end), ...
        layer_size(iter+1), layer_size(iter)+1);
end

aa{1}=x;
% Do forward propagation
for iter =1:layer_num-1
    % ones(1, m) is for bias
    aa{iter} = [ones(1, M); aa{iter}];
    %%%----- z[n]=W[n]a[n-1]  ------%%%
    zz{iter}=ww{iter}*aa{iter};
    %%%----- a[n]=g(z[n])------%%%
    if iter ==layer_num-1 %output layer has to use sigmoid
        [aa{iter+1},~] = activation(zz{iter},3);
    else
        [aa{iter+1},~] = activation(zz{iter},act_option);
    end
end
tp=aa{layer_num};

[~, pred] = max(tp, [], 1); pred=pred-1;
[~, pobs] = max(t, [], 1); pobs=pobs-1;

titleNN=[];
for iter =1:layer_num %%%----- N (in book) ------%%%
    titleNN=[titleNN,' ',num2str(layer_size(iter))];
end

ws1=15; ws2=18;
subplot(131);plot(res(2:end-1));
ylabel('Avg Misfit');xlabel('Iteration #');set(gca,'FontSize', ws1);
title({['(a) NN convergence plot '];[ 'Total layer number =',num2str(layer_num)]},'FontSize', ws2);
subplot(132);plot([1:M],pobs,'r*')
ylabel('Example Class');xlabel('Example #');set(gca,'FontSize', ws1);
title({['(c) NN result (Obs=red, Pred=green) ']; [ 'NN struture =',titleNN]},'FontSize', ws2);
subplot(133);plot([1:M],pred,'g*')
ylabel('Example Class');xlabel('Example #');set(gca,'FontSize', ws1);
title({['(c) NN result (Obs=red, Pred=green) ']; [ 'NN struture =',titleNN]},'FontSize', ws2);

% ws1=15; ws2=18;
% subplot(121);plot([1:M],pobs,'r*')
% ylabel('Example Class');xlabel('Example #');set(gca,'FontSize', ws1);
% title({['(c) NN result (Obs=red, Pred=green) ']; [ 'NN struture =',titleNN]},'FontSize', ws2);
% subplot(122);plot([1:M],pred,'g*')
% ylabel('Example Class');xlabel('Example #');set(gca,'FontSize', ws1);
% title({['(c) NN result (Obs=red, Pred=green) ']; [ 'NN struture =',titleNN]},'FontSize', ws2);

% subplot(122);plot([1:M],pobs,'r*',[1:M],pred,'g*')
% ylabel('Example Class');xlabel('Example #')
% title({['(c) NN result (Obs=red, Pred=green) ']; [ 'NN struture =',titleNN]});

