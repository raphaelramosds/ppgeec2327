function [x,t,alpha,lim,ww,layer_size,obj,act]=Datain(M,N,x,t)

for i=1:M
    x(:,i)=round(rand(N,1));
end
for i=1:M
    if sum(x(:,i))==0;t(i)=1;end
end

% Now we are seting about 1/2
% of x(:,:) to have no 1 in the Nx1 vector
% so about 1/2 data are classed at t=1
t0=find(t==0); t1=find(t==1);
x(:,t0(1:round(M/2)-numel(t1)))=0.0;
t(t0(1:round(M/2)-numel(t1)))=1;
rank = randperm(M);
x=x(:,rank);
t=t(rank);

alpha=1.0; % step size
lim=.00001; % stopping criterion
layer_size=[N,10,1]; % first and last elements are the input and output layer size
layer_size=[N,10,1];
obj=1; %objective function, 1 for L2 norm, 2 for likelihood
act=1; %active function, 1 for sigmoid norm, 2 for ReLU

layer_num=numel(layer_size); %layer number include the input and output
ww=cell(layer_num-1,1); 
for ilayer=1:layer_num-1
    %  initialize the weights of the neural network with random number
    ww{ilayer}=rand(layer_size(ilayer+1),layer_size(ilayer)+1)*0.1; % +1 for bias
end

end