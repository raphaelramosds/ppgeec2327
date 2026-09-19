function [grad,res]=gradientnn(M,x,t,ww,layer_size,obj,act)
% output:
%  res - objective function value
%  grad - (objective function respective to weights)
%
% Input:
% M   - # data examples
%(x,t)- Training data pairs
%ww- filter for all layers
% t   - # observed data
% layer_size - indicate nn structure
%obj=1 for L2, 2 for likelihood
%act=1 for sigmoid, 2 for ReLU

%% Setup some useful variables
layer_num=numel(layer_size); %layer number include the input and output
aa=cell(layer_num-1,1);
zz=cell(layer_num-1,1);
aa{1}=x; %first layer is the input layer
penalize=0;
%% Part 1: Feedforward the neural network and return the cost in the variable res.
% Do forward propagation
for iter =1:layer_num-1 
    aa{iter} = [ones(1, M); aa{iter}]; % z[n]=W[n]a[n-1], 1 is for bias
    zz{iter}=ww{iter}*aa{iter}; % a[n]=g(z[n])
    if iter ==layer_num-1 %output layer use sigmoid
        [aa{iter+1},~] = activation(zz{iter},1);
    else
        [aa{iter+1},~] = activation(zz{iter},act);
    end
    penalize =penalize+ sum(sum(ww{iter}.^ 2)); % add regularization 
end
t_pred = aa{layer_num};
[res,in]= misfit( t_pred,t,1/M,obj); %obj=1 for L2, 2 for likelihood
%% Part 2: Implement the backpropagation algorithm to compute the gradients.
% Implement backpropagation
for iter=layer_num-1:-1:1
    if iter ==layer_num-1 %output layer has to use sigmoid
        [~,dg]=activation(zz{iter},1);
    else
        [~,dg]=activation(zz{iter},act);  %in=dg[i].*in 
    end
    in=dg.*in; % in is the backward field
    grad{iter}=in*aa{iter}'; % de(j,k)=in*(a[i-2])T
    in=ww{iter}'*in;  % in=W'[i]*in
    in = in(2:end, :);
end
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

function [ J,in ] = misfit( Y_pred,Y_obs,scale,type)
%misfit function
%input:
%Y_pred: predicted value
%Y_obs: observed vakue
%scale: scale the misift
%type: objetive function type: 1 for L2, 2 for likelihood
%output:
% J is the value of the objective function
% in is the gradient of J with respective to Y_pred

if type==1 % for L2 norm objective function
    %J = sqrt(sum( sum( (Y_pred-Y_obs).^2 ) ));  
    J = sum( sum( (Y_pred-Y_obs).^2 ) );  
    in = 2*(Y_pred-Y_obs);
elseif type==2 % for likelihood  objective function
    J = sum(sum(-Y_obs.*log(Y_pred) - (1-Y_obs).*log(1-Y_pred)));
    in = -Y_obs./Y_pred + (1-Y_obs)./(1-Y_pred);  % dJ / d Y_pred
end
J=J*scale;in=in*scale;
end
