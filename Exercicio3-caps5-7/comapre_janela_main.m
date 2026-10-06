clear
clc
close all

%%%%%%%%%%%%%%Comparision of ML methods in Identifyting migration artifacts
%%%%%%%%%%%%%%and reflector points%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------------%
%                       Build the training sets                          %
%------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Loading migration image and features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load 'mig_and_features.mat'
[nz,nx]=size(mig);          % get the size of the model

%%%%The loaded data has four parts
%%%%mig    :: migration image with reflectors and artifacts
%%%%cohe   :: coherency feature
%%%%angl   :: local angle feature
%%%%energy :: Amplitude feature

%%%%%%%% Recale features by [x-mean(x)]/[max(x)-min(x)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Rescale the coherency
cohe_vec=reshape(cohe,1,nz*nx);           % Reshape the 2D features to 1D
cohe_mean=mean(cohe_vec);                 % Compute the mean of the features
cohe_std=max(cohe_vec)-min(cohe_vec);     % Compute difference between the maximum and minimum value of the feature
cohe_vec2=(cohe_vec-cohe_mean)./cohe_std; % Feature scaling

%Rescale the angle (same as above)
angl_vec=reshape(angl,1,nz*nx);
angl_mean=mean(angl_vec);
angl_std=max(angl_vec)-min(angl_vec);
angl_vec2=(angl_vec-angl_mean)./angl_std;

%Rescale the energy (same as above)
engy_vec=reshape(engy,1,nz*nx);
engy_mean=mean(engy_vec);
engy_std=max(engy_vec)-min(engy_vec);
engy_vec2=(engy_vec-engy_mean)./engy_std;

% reshape the scaled 1D feature to 2D
cohe2=reshape(cohe_vec2,nz,nx);
angl2=reshape(angl_vec2,nz,nx);
engy2=reshape(engy_vec2,nz,nx);

% Feature-domain plots (coherency/angle/amplitude) removed

%%%%%%% Build Training Sets
%%%%%%%%%%%%%%%%%%%%%%%%%%%

load 'xzpicks_50artif_50ref.mat'
n_arti = size(x, 1);
n_refl = size(x2, 1);
n_total=n_refl+n_arti;     % compute the total number of training examples

% parametros de janelamento
winx = 5;
winz = 5;

% parametros de treinamento (ajuste aqui)
maxiter_logreg = 800;
maxiter_ann = 800;

X=zeros(n_total,winx * winz); % define the training dataset, which is (n_total by winx * winz). 
                    % n_total : the number of training examples
                    % winx : width of window
                    % winz : height of window
for iz=1:n_refl     % build the training set for reflector points
    X=montar_janela(X, iz, iz, x2, z2, mig, winx, winz);
end 

kk=1;
for iz=n_refl+1:n_total   % build the training set for artifact points
    X=montar_janela(X, iz, kk, x, z, mig, winx, winz);
    kk=kk+1;
end
Y=zeros(n_total,1); Y(1:n_refl)=1;Y(n_refl+1:end)=0; % build the training set for outputs.
                                                     % Y(i) = 1 means this example is a reflector point
                                                     % Y(i) = 0 means this example is a artifact point

% Normalize window features using training-set statistics
X_mean = mean(X,1);
X_scale = max(X,[],1) - min(X,[],1);
X_scale(X_scale==0) = 1;
X = (X - X_mean) ./ X_scale;






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------------%
%                           Logistic Regression                          %
%------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% Make decesion boundary by logistic regression %%%%%%%%%%%%
[m,n]=size(X);     % Get the size of training example
X2=[ones(m,1) X];  % Add intercept term to the training set

disp('Calculating decision boundarys')
maxiter=maxiter_logreg;  % Iteration numbers
theta=rand(n+1,1)-0.5;   % Intialize the model parameters
step=10;                 % Set the step length

for iter=1:maxiter
    [cost, grad] = costFunction2(theta, X2, Y);          % Compute data misfit (cost) and gradient (grad)
    if(mod(iter,20)==0)                                  % Show the data misfit at every 20 iterations
      fprintf('iter= %d res = %f\n',iter,cost)
    end
    theta2=theta-step.*grad;                             % Update the model parameters with a trail step length
    [cost1, grad_temp] = costFunction2(theta2, X2, Y);   % Compute the new misfit (cost1) and gradient (grad_temp)
  %  fprintf('res1 = %f\n',cost1)
  
    while(cost1>cost)  % If the new misfit is larger, reduce step length and try again
       step=step*0.8;  % Reduce the step length
       theta2=theta-step.*grad;   % Update the model parameters with reduced trail step length
       [cost1, grad_temp] = costFunction2(theta2, X2, Y);  % Compute the new misfit
       fprintf('step = %f, res1 = %f\n',step,cost1)        % print the step length and data misfit
    end    % When new misfit gets smaller, update model parameters
    
    theta=theta2; 
end

%{
%%%%%%%%%%% 3D Plot of the decision boundary (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Display decision boundarys')
[X_surf1,Y_surf1]=meshgrid(min(cohe_vec2):0.01:max(cohe_vec2),min(angl_vec2):0.01:max(cohe_vec2));
Z_surf1=-(theta(1)+theta(2).*X_surf1+theta(3).*Y_surf1)./theta(4);  % compute the hyperplane predicted by the logistic regression
hold on
surf(X_surf1,Y_surf1,Z_surf1)  % display the hyperplane

%%%%% Use the decesion boundary to decide the all the imaging point 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

disp('Classify every image point in the migration image')

mig_result1=zeros(size(mig));  % mig_result1 has the same size as the migration image will save the predicted result
for ix=1:nx
    for iz=1:nz
     janela = montar_janela(mig, ix, iz, winx, winz);
     janela = (janela - X_mean) ./ X_scale;
     p = sigmoid([1 janela] * theta);
     mig_result1(iz,ix) = p > 0.5;
    end
end
figure(4);imagesc(mig_result1)
title('Classified Results')




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------------%
%                            Neural Network                              %
%------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%Nonliear Neural Wrok%%%%%%%%%%%%%%%%%%%%%%%%%
%In this section, we will create a 2 layer neural work (one hidden layer 
%and one output layer). Also, to increase the nonliearity, we increase the 
%freedom of input features from 1 to 2. (x1,x2,x3) --> (x1,x2,x3,x1^2,x2^2,x3^3,x1x2,x1x3,x2x3)
%If we increase the input features from 1 to 3, then we have 
%(x1,x2,x3) --> (x1,x2,x3,x1^2,x2^2,x3^2,x1^3,x2^3,x3^3,...)

%%%%%% Increase the freedom of the input features in the trainig set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dimen=10;    
X2 = [ones(m,1) X];                    % each row: [1, flattened window]
input_layer_size = size(X2,2)-1;       % number of input units (window size)
hidden_layer_size = max(5, min(50, round(input_layer_size/2))); % heuristic
num_labels = 1; 

%%%%%%% Set the number of unit in each layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m1,n1]=size(X2);                 % Get the size of training set
input_layer_size = size(X2,2)-1;  % Set the number of node in the input layers
hidden_layer_size = max(5, min(50, round(input_layer_size/2)));  % heuristic for number of hidden nodes
num_labels = 1;                   % Set the number of node in the output layers

%%%%%%% Initialize the model parameters in each layer by random
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

initial_theta1=(rand(input_layer_size+1,hidden_layer_size)-0.5)';
initial_theta2=(rand(hidden_layer_size+1,num_labels)-0.5)';

% Unroll parameters to vector
nn_params = [initial_theta1(:) ; initial_theta2(:)];

%%%%%%% Training (same workflow as logistic regression)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lamda=0.0; step=10; maxiter=maxiter_ann;  % lamda: regularization term (higher lamda can avoid over-fitting but will lost accuracy)
for iter=1:maxiter
    [cost,grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...  % nnCostFunction is the kernel used to compute the data misfit and gradient
                    num_labels, X2, Y, lamda); 
     if(mod(iter,50)==0||iter==1)                         
        fprintf('iter= %d, res = %f\n',iter,cost)
     end
     nn_params2=nn_params-step*grad; 
     [cost1,grad_temp] = nnCostFunction(nn_params2, input_layer_size, hidden_layer_size, ...
                    num_labels, X2, Y, lamda); 
     while(cost1>cost)
        step=step*0.8;
        nn_params2=nn_params-step*grad; 
        [cost1,grad_temp] = nnCostFunction(nn_params2, input_layer_size, hidden_layer_size, ...
                    num_labels, X2, Y, lamda);   
        fprintf('step = %f, res1 = %f\n',step,cost1)                      
     end
     nn_params=nn_params2;
end

%% 3D feature-domain plot removed

%%%%%%%%%%%%% 3D Plot of the decision boundary (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dx=0.01;
xx1=min(cohe_vec2):dx:max(cohe_vec2);
xx2=min(angl_vec2):dx:max(angl_vec2);
xx3=min(engy_vec2):dx:max(engy_vec2);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% vals=visualizeBoundaryNN3D([xx1' xx2' xx3'], Theta1, Theta2,dx);  % skipped: visualization expects 3 features
% Visualization of NN decision boundary skipped (expects 3-feature input).

%%%%%% Prediction. The trained model parameters are saved in Theta1 and Theta2. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mig_result2 = zeros(size(mig));
for ix=1:nx
  for iz=1:nz
    janela = montar_janela(mig, ix, iz, winx, winz);
    janela = (janela - X_mean) ./ X_scale;
    val = [1 janela];
    h1 = sigmoid(val * Theta1');
    h2 = sigmoid([1 h1] * Theta2');
    mig_result2(iz,ix) = h2 > 0.5;
  end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------------%
%                        Support Vector Machine                          %
%------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% For your homework you will need to re-build the training set X and Y. 
%%%% Do it here... 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Support Vector Machine%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Make decesion boundary by SVM
C = 1;              % The C parameter is a positive value that controls the
                    % penalty for misclassified training examples. A large 
                    % C parameter tells the SVM to try to classify all the 
                    % examples correctly. C plays a role similar to
                    % 1/lamda, where lamda is the regularization parameter                   
sigma = 0.1;        % sigma controls the shape of gaussian function. In other
                    % words, how fast the similarity metric decrease

% "svmTrain" is the matlab function used for SVM training. Here we choose
% Gaussian as the kernel. If you want to use the linear kernel, please use
% model= svmTrain(X, Y, C, @linearKernel); 
model= svmTrain(X, Y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));  
                                                                    
                                                                  
% 3D feature-domain plot removed

%%%%%%%%%%%%% 3D plot of the decision boundary (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization of SVM decision boundary skipped (model trained on flattened windows)
% visualizeBoundary3D expects 3 input features and will fail for high-dimensional inputs.
% Skipping this 3D visualization to avoid dimensionality errors.

%%%%%%%%%%% Prediction. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bb=model.b;  % model.b is the trained model parameters
ww=model.w;  % model.w is the trained bias term
mig_result3=zeros(size(mig));
for ix=1:nx
    for iz=1:nz
        janela = montar_janela(mig, ix, iz, winx, winz);
        janela = (janela - X_mean) ./ X_scale;
        val2 = svmPredict2(model, janela);
        mig_result3(iz,ix) = val2;
    end
end

%%%%%%%%%%%% Plot all (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;subplot(1,3,1);imagesc(mig_result1);title('Logistic Regression')
subplot(1,3,2);imagesc(mig_result2);title('Neural Network')
subplot(1,3,3);imagesc(mig_result3);title('SVM')

% Final 3D feature-domain comparison plot removed
