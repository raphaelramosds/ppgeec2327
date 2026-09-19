function Display1(M,res,nit,ww,layer_size,x,t,obj,act,thresh)
tp=t*0;tpp=tp;fs=7;fs1=fs+1;fst=fs;
%thresh=.34;
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
title('a) RMS Error vs Iter. #','fontsize',fs1);

if obj==2;text(nit/6,max(res)*.9,'Cross-entropy NN','fontsize',fst);
else;
text(nit/6,max(res)*.9,'Least-squares NN','fontsize',fst);end

text(nit/15,max(res)*.95,...
    ['# nodes/layer = [',num2str(layer_size),']'],'fontsize',fst)

if act==2;text(nit/6,max(res)*.85,...
        'ReLU activation','fontsize',fst);else;
text(nit/6,max(res)*.85,...
        'Sigmoid activation','fontsize',fst);end

ylabel('RMS Misfit');xlabel('Iteration #','fontsize',fs1)
subplot(132);plot([1:M],t,'r*',[1:M],tp,'go')
ylabel('Example Class','fontsize',fs1);xlabel('Example #','fontsize',fs1)
title('b) (Obs=red, Pred=green)','fontsize',fs1)
subplot(133);plot([1:M],t,'r*',[1:M],tpp,'g*')
ylabel('Example Class','fontsize',fs1);xlabel('Example #','fontsize',fs1)
title(['c) Threshold Value = ',num2str(thresh)],'fontsize',fs1)
text(M/9,.5,'Red * = Mistake','color','red','fontsize',fs)
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
