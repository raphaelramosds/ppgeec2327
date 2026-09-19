function [res,grad_params]=gradientnn_fmincg(nn_params,M,x,t,layer_size,obj_option,act_option, lambda);
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
%obj_option=1 for L2, 2 for likelihood
%act_option=1 for sigmoid, 2 for ReLU

% Part 1: Feedforward the neural network and return the cost in the
%         variable res.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients.
%         You should return the partial derivatives of the cost function with
%         respect weight, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K.

%% Setup some useful variables
layer_num=numel(layer_size); %layer number include the input and output
aa{1}=x; %first layer is the input layer
penalize=0;

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

%% Part 1: Feedforward the neural network and return the cost in the variable res.

% Do forward propagation
for iter =1:layer_num-1 %%%----- N (in book) ------%%%
    % ones(1, m) is for bias
    aa{iter} = [ones(1, M); aa{iter}];
    %%%----- z[n]=W[n]a[n-1]  ------%%%
    zz{iter}=ww{iter}*aa{iter};
    %%%----- a[n]=g(z[n])------%%%
    if iter ==layer_num-1 %output layer use softmax
        [aa{iter+1},~] = activation(zz{iter},3);
    else
        [aa{iter+1},~] = activation(zz{iter},act_option);
    end
    %Add some regularization
    penalize =penalize+ sum(sum(ww{iter}.^ 2)); % include regularization
end
% final output for forward propagation Ypred
t_pred = aa{layer_num};

% calculate the objective function and the dirivative of objective function with respect to t_pred using likelihood or L2 type
%obj_option=1 for L2, 2 for likelihood
[res,in]= misfit( t_pred,t,1/M,obj_option);

% add regularization
res = res + (lambda/(2*M)) * penalize;
%% Part 2: Implement the backpropagation algorithm to compute the gradients.
%  writen according to Backpropagation Operation

% Implement backpropagation
for iter=layer_num-1:-1:1
    %'sigmoidgrad' means compute the gradient of the sigmoid function
    if iter ==layer_num-1 %output layer has to use sigmoid
        [~,dg]=activation(zz{iter},3);
    else
        [~,dg]=activation(zz{iter},act_option);  %%%----- in=dg[i].*in (in book) ------%%%
    end
    if iter ==layer_num-1 %output layer use softmax
        for i=1:M
            in(:,i)=dg(:,:,i)'*in(:,i);
        end
    else
        in=dg.*in;
    end
    % in is the backward field, A is the forward field
    grad{iter}=in*aa{iter}'+ (lambda/M) * ww{iter}; %%%----- de(j,k)=in*(a[i-2])T (in book) ------%%%
    % update in for calculation of the gradient with respect to next weights
    in=ww{iter}'*in;  %%%----- in=W'[i]*in (in book) ------%%%
    in = in(2:end, :);
end

% store all the weights into one big vector
grad_params=[];
for iter=1:layer_num-1
    grad_params=[grad_params;grad{iter}(:)];
end

end