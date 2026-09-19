function [ g,dg ] = activation( z,type )
%activation function
%
%input:
%z: input for atviation function
%type: objetive function type: 1 for sigmoid, 2 for ReLU
%
%output:
% g is the value of the activation function
% dg is the gradient of g with respective to z

if type==1 % for sigmoid
    g = 1.0 ./ (1.0 + exp(-z));
    dg = g .* (1 - g); %Compute the gradient of the sigmoid function
% elseif type==2 % for ReLU
%     g=z*0.0; dg=z*0.0;
%     g(z>=0)=z(z>=0);
%     g(z<0)=z(z<0)*0.0;
%     dg(z>=0)=1.0;
%     dg(z<0)=0.0;
elseif type==2 % for Exponential ReLU:https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html
    g=z*0.0; dg=z*0.0;
    g(z>=0)=z(z>=0);
    g(z<0)=(exp(z(z<0))-1);
    dg(z>=0)=1.0;
    dg(z<0)=exp(z(z<0));
elseif type==3 % for Softmax
    [n1,n2]=size(z);
    ez=exp(z);
    ezsum=sum(ez,1);
    g=ez./repmat(ezsum,[n1,1]);
    dg=zeros(n1,n1,n2);
    for i=1:n2
        dg(:,:,i)=diag(g(:,i))-g(:,i)*g(:,i)';
    end
else
    display('You entered the wrong type for the activation function');
    stop
end


end

