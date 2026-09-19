%% Simple Neural Network LAB
% Learn how to to do forward and backward propogation according to chsingle.pdf by Gerard Schuster
% coded by Zongcai Feng and Gerard Schuster

% This is the main program to run Neural Network
% It involves function
%  gradientnn.m - Neural network cost function and gradient calculation
%  misfit.m - Objective function and gradient with repected to predicted output
%              (currently have likelihood and L2)
%  activation.m - activation function and gradient with repected to Z (currently use sigmoid and ReLU)
%  Displaynn.m - predict the output and plot

%% ================= Set up input =================
clear ; close all; clc

% x(N,M)  - input- M input feature vectors with size Nx1
mn=50;  % number of input for each class
N=5;    % # of unknowns (w0, w1, wN-1)
M=mn*(N+1);  % # of equations constraint
x=zeros(N,M); % input
t=zeros(N+1,M); % labels

for in=0:N
    x_temp=[ones(in,1);zeros(N-in,1)];
    for i=1:mn
        rankmn = randperm(N);
        x(:,in*mn+i)=x_temp(rankmn);
        t(in+1,in*mn+i)=1;
    end
end

% Randomly select 100 data points to display
sel = randperm(M);
x=x(:,sel); t=t(:,sel);
figure(1);
subplot(121);displayData(x(:, 1:100)',N); 
title({['(a) Data display'];[' white indicates 1, grey indicates 0']},'FontSize', 20);
subplot(122);displayData(t(:, 1:100)',N+1);
title({['(b) Label display'];['location of white indicate the class']},'FontSize', 20);
%% ================= Set up parameters for NN structure=================--------------------------------------------------------------------------------
display('-----------------------------------------------------------Instruction of NN structure---------------------------------------------------------------------------------------');
display('The NN code requires the input the number of nodes in each layer.');
display('(The layers defines in this NN structure do not include the layers for input feathuires and no layers for output labels)');
display('(E.g:[20,15,10], three hidden layers and their nodes numbers are 20,15,10, respectively)');
display('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ');
layer_temp=input('Please input the number of nodes in each layer:  ');
display(' ');
obj_option=input('Please input the objective function type, 1 for L2 norm, 2 for likelihood:   ');
display(' ');
act_option=input('Please input the active function type, 1 for sigmoid norm, 2 for ReLU:   ');
display('For lager layers numbers (>=6), please choose sigmoid norm for robust convergence');
display(' ');
display('200 iterations should be enougth for hidden layers<=3, more layers requires more iterations');
iteration_num=input('Please input iteration numember:   ');

layer_size=[N,layer_temp,N+1]; % layer_size include the input and output layer
layer_num=numel(layer_size); %layer number include the input and output
%% ================ Initializing Pameters for Training ================
%  initialize the weights of the neural network with random number
nn_params=[];
for ilayer=1:layer_num-1
    ww{ilayer}=rand(layer_size(ilayer+1),layer_size(ilayer)+1)*0.1;
    % store all the weights into one big vector
    nn_params=[nn_params;ww{ilayer}(:)];
end
lambda = 0.0; %for regularization, if add regularization, suggest lambda<0.01
%% =================== Training NN ===================
%  To train your neural network, we will now use steepest decent and line search
fprintf('\nTraining Neural Network... \n')

% Create "short hand" for the cost function to be minimized
costFunction = @(p) gradientnn_fmincg(p,M,x,t,layer_size,obj_option,act_option,lambda);

% fmincg use Polack-Ribiere flavour of conjugate gradients for steplengh
[nn_params, cost] = fmincg(costFunction, nn_params, iteration_num); %line search

%% ================= Implement Predict =================
%  After training the neural network, you will now implement the "Displaynn" function to use the
%  neural network to predict the labels of the training set.

x(:,sel)=x; t(:,sel)=t;
figure(2);Displaynn_multiclass(M,cost,nn_params,layer_size,x,t,act_option);pause(0.5)
