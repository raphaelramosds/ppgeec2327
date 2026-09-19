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
nit=400;res=zeros(nit,1);
layer_num=numel(layer_size); %layer number include the input and output

% ---------Check gradient without regularization--------

for obj=1:2
    for act=1:2
        
       % disp(['obj = ',num2str(obj),', act = ',num2str(act),]);
        
        [grad,~]=gradientnn(M,x,t,ww,layer_size,obj,act);
        
        grad_numel=grad;
        ww_numel=ww;
        
        e = 1e-4;
        for ilayer=1:layer_num-1
            grad_numel{ilayer}=grad{ilayer}*0.0;
            for irow=1:layer_size(ilayer+1)
                for icolumn=1:layer_size(ilayer)+1
                    ww_numel=ww;
                    ww_numel{ilayer}(irow,icolumn)=ww{ilayer}(irow,icolumn)-e;
                    [~,res1]=gradientnn(M,x,t,ww_numel,layer_size,obj,act);
                    ww_numel{ilayer}(irow,icolumn)=ww{ilayer}(irow,icolumn)+e;
                    [~,res2]=gradientnn(M,x,t,ww_numel,layer_size,obj,act);
                    grad_numel{ilayer}(irow,icolumn) = (res2 - res1) / (2*e); %numeical gradient
                    
                    disp(['Analytic grad = ',num2str(grad{ilayer}(irow,icolumn)),...
                        ', Nume grad = ',num2str(grad_numel{ilayer}(irow,icolumn)), ...
                        ', Diff = ',num2str(grad{ilayer}(irow,icolumn)-grad_numel{ilayer}(irow,icolumn))]);
                end
            end
        end
        
    end
end

 diffgrad1=(grad{1}(:,:)-grad_numel{1}(:,:))/max(abs(grad{1}(:)));
  diffgrad2=(grad{2}(:,:)-grad_numel{2}(:,:))/max(abs(grad{2}(:)));
  figure(2);
  subplot(121);imagesc( diffgrad1);colorbar;
  subplot(122);imagesc( diffgrad2);colorbar;
  figure(1)

%% ------------------------------------------
ww_ori=ww;

nit=200;

for obj=1:2
    for act=1:2
        
        ww_old=ww_ori;
        ww=ww_ori;
        
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
             
        res=res/max(res);
        sub_num=221+(obj-1)*2+(act-1);
        figure(1);subplot(sub_num);
        subplot(211);% This is Jerry's addition
        plot(res(1:nit));
       % title({[' RMS Error vs Iter. #'];title_name1;title_name2})

        ylabel('RMS Misfit');xlabel('Iteration #');
        ylim([0.0,1.0]);xlim([1,nit]);
        if act==1 & obj==1;title(['a) RMS Error vs Iter. #'])
            text(25,.4,['Sigmoid Activation'],'fontsize',9);
            text(25,.5,['L2 Objective'],'fontsize',9);
        end
        if act==1 & obj==2; title(['c) RMS Error vs Iter. #'])
            text(25,.4,['Sigmoid Activation'],'fontsize',9);
            text(25,.5,['X-entropy Objective'],'fontsize',9);
        end
        if act==2 & obj==1;  title(['b) RMS Error vs Iter. #'])
            text(25,.4,['ReLU Activation'],'fontsize',9);
            text(25,.5,['L2 Objective'],'fontsize',9);
        end
        if act==2 & obj==2; subplot(211);
            title(['a) RMS Error vs Iter. # : [',num2str(layer_size),']'])
            text(25,.4,['ReLU Activation'],'fontsize',9);
            text(25,.5,['X-entropy Objective'],'fontsize',9);
        end
        
    end
end
  subplot(223);imagesc( diffgrad1);colorbar;title('b) First layer: \Delta\partial\epsilon/\partialw_{ij}')
  xlabel('Column index');ylabel('Row index')
  subplot(224);imagesc( diffgrad2);colorbar;title('c) Second layer: \Delta\partial\epsilon/\partialw_{ij}')
    xlabel('Column index');ylabel('Row index')
    print -depsc INN.GradDiff.eps