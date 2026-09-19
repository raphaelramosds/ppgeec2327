function Display(M,res,nit,ww,layer_size,x,t,obj,act)
tp=t*0;tpp=tp;
thresh=.34;
layer_num=numel(layer_size); %layer number include the input and output
aa=cell(layer_num-1,1);
zz=cell(layer_num-1,1);
aa{1}=x;
for iter =1:layer_num-1
    aa{iter} = [ones(1, M); aa{iter}];
    zz{iter}=ww{iter}*aa{iter};
    if iter ==layer_num-1
        [aa{iter+1},~] = activation(zz{iter},1);
    else
        [aa{iter+1},~] = activation(zz{iter},act);
    end
end
tp=aa{layer_num};
for m=1:M
    if tp(m)>=thresh;tpp(m)=1;end
    if tp(m)<thresh;tpp(m)=0;end
end
subplot(131);plot(res(1:nit));
title('a) RMS Error vs Iter. #');

if obj==2;text(nit/6,max(res)*.9,'Cross-entropy NN','fontsize',8);
else;
text(nit/6,max(res)*.9,'Least-squares NN','fontsize',8);end

text(nit/15,max(res)*.95,...
    ['# nodes/layer = [',num2str(layer_size),']'],'fontsize',8)

if act==2;text(nit/6,max(res)*.85,...
        'ReLU activation','fontsize',8);else;
text(nit/6,max(res)*.85,...
        'Sigmoid activation','fontsize',8);end

ylabel('RMS Misfit');xlabel('Iteration #')
subplot(132);plot([1:M],t,'r*',[1:M],tp,'go')
ylabel('Example Class');xlabel('Example #')
title('b) (Obs=red, Pred=green)')
subplot(133);plot([1:M],t,'r*',[1:M],tpp,'g*')
ylabel('Example Class');xlabel('Example #')
title(['c) Threshold Value = ',num2str(thresh)])
text(M/9,.5,'Red * = Mistake','color','red')
end

function [ g,dg ] = activation( z,type )
%activation function
%input:
%z: input for atviation function
%type: objetive function type: 1 for sigmoid, 2 for ReLU
%output:
% g is the value of the activation function
% dg is the gradient of g with respective to z

if type==1 % for sigmoid
    g = 1.0 ./ (1.0 + exp(-z));
    dg = g .* (1 - g); %Compute the gradient 
elseif type==2 % for ReLU
    g=z*0.0; dg=z*0.0;
    g(z>0)=z(z>0);
    g(z<=0)=z(z<=0)*0.0;
    dg(z>0)=1.0;
    dg(z<=0)=0.0;
end
end
