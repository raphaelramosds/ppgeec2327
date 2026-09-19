% x(N,M)- input- M input feature vectors with size Nx1
% t(1,M) - input- M input target vectors with size 1x1
% t= 1 or 0
%layer_size- input- # of layers
%obj- input- 1=l2 & 2=x-entropy
%act- input- 1=sigmoid & 2=ReLU
% lambda - reg. param
% ww - input- random starting Nx1 weight vector for each node
M=100; % # of equations constraint
N=5; % # of unknowns (w1, w2, w3, w4, w5) for 1st node !modified by Zongcai
x=zeros(N,M);t=zeros(1,M);
[x,t,alpha,lim,ww,layer_size,obj,act]=Datain(M,N,x,t); % Create training data
act=1;
obj=1;
nit=400;res=zeros(nit,1);
layer_num=numel(layer_size); %layer number include the input and output

%% ------------------------------------------
x(:,1:50)=zeros(5,50);
ww_old=ww;
for k=1:nit % Looping over iterations
    alpha=1;    % step size (orig, case: 1)
    [grad,res(k)]=gradientnn(M,x,t,ww_old,layer_size,obj,act);
    for ilayer=1:layer_num-1
        ww{ilayer}=ww_old{ilayer}-alpha*grad{ilayer};
    end
    % Bisection line search for step length
    [~,res1]=gradientnn(M,x,t,ww,layer_size,obj,act);
    while (res1>res(k)) && (alpha>lim)
        alpha=alpha*0.5;
        for ilayer=1:layer_num-1
            ww{ilayer}=ww_old{ilayer}-alpha*grad{ilayer};
        end
        [~,res1]=gradientnn(M,x,t,ww,layer_size,obj,act);
    end
    ww_old = ww;  %%% update successful
end
Display(M,res,nit,ww,layer_size,x,t,obj,act)
print -depsc Fig1.balanced.L2.sig.eps