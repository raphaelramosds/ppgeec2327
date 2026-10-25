clear all,close all

addpath ./CNN
addpath ./util
addpath ./data/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Inputs                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ntrain=200; % The number of training set per class
nclass=3;   % Number of classes: 1-background, 2-small_crack, 3-big_crack
class_names={'background','small\_crack','big\_crack'};

wdx=8; 
wdz=8;
image_x=wdx*2+1; % Original Image Patch Size X
image_z=wdz*2+1; % Original Image Patch Size Z
% Intepolate image to make sure size before pooling is even
image_x_new=32;  % Intepolated Image Patch Size X
image_z_new=32;  % Intepolated Image Patch Size Z

opts.alpha = 1;   % Learning Rate
opts.batchsize = 20;  % Minibatch Size
opts.numepochs = 10;  % Iteration Number

%%%%  CNN structure %%%%%%%
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer   
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer   
};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Step 1 Prepare test and training set            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load image_input.mat 

%%%%%%%%%%%% Load Prepared training set%%%%%%%%%%%%% 
% Multi-class training set: 3*ntrain = 600 points
%   1:ntrain            -> background  (label [1;0;0])
%   ntrain+1:2*ntrain   -> small_crack (label [0;1;0])
%   2*ntrain+1:3*ntrain -> big_crack   (label [0;0;1])
% If the file does not exist yet, pick the points manually and save it.
if exist('./data/multiclass_train.mat','file')
    load multiclass_train.mat
else
    train.label=zeros(nclass,nclass*ntrain);
    train.trainx=zeros(1,nclass*ntrain);
    train.trainz=zeros(1,nclass*ntrain);
    for ic=1:nclass
        idx=(ic-1)*ntrain+1:ic*ntrain;
        figure(ic)
        imagesc(image_input);
        title(['Pick ',class_names{ic},' Points (',num2str(ntrain),')'],'fontsize',22);
        train.label(ic,idx)=1;
        [px,pz]=ginput(ntrain);
        % Patch indexes must be integers inside the image
        train.trainx(idx)=min(max(round(px),1),size(image_input,2));
        train.trainz(idx)=min(max(round(pz),1),size(image_input,1));
        pause(1)
        close(ic)
    end
    save('./data/multiclass_train.mat','train');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Step 2 Convert Image into Image patches             % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nz,nx]=size(image_input);
image_pad=padimage(image_input,wdz,wdx);
%%% Obtain image batches from full image
image_slice=zeros(image_z,image_x,nz*nx);
image_slice_new=zeros(image_z_new,image_x_new,nz*nx);
[Xq,Zq]=meshgrid(1:(image_x-1)/(image_x_new-1):image_x,1:(image_z-1)/(image_z_new-1):image_z);
ii=1;
for ix=1:nx
    for iz=1:nz
        image_slice(1:image_z,1:image_x,ii)=image_pad(iz:iz+image_z-1,ix:ix+image_x-1);
        image_slice_new(:,:,ii)=interp2(image_slice(:,:,ii),Xq,Zq);
        ii=ii+1;
    end
end

%%% Obtain image batches from train set
train_slice=zeros(image_z,image_x,nclass*ntrain);
train_slice_new=zeros(image_z_new,image_x_new,nclass*ntrain);
[Xq,Zq]=meshgrid(1:(image_x-1)/(image_x_new-1):image_x,1:(image_z-1)/(image_z_new-1):image_z);
for in=1:nclass*ntrain
        train_slice(1:image_z,1:image_x,in)=...
            image_pad(train.trainz(in):train.trainz(in)+image_z-1,train.trainx(in):train.trainx(in)+image_x-1);
        train_slice_new(:,:,in)=interp2(train_slice(:,:,in),Xq,Zq);
end

clear image_slice train_slice

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Step 3 Train CNN structure                    % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Input Training and Test set
train_x =train_slice_new;  % Training set
train_y =train.label;      % Training label (nclass x nclass*ntrain)
test_x =image_slice_new;   % Test set


rand('state',0)
cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Step 4 Input full image for Classification             % 
%   (Seperate The Image into 10 parts In case of out of memory)  %  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predicted class = index of the largest output neuron (1, 2 or 3).
% un = ll-1 -> 0: background, 1: small_crack, 2: big_crack

for ii=1:9
disp(['Part ',num2str(ii),' Start']);
uu = cnnff(cnn, test_x(:,:,457*100*(ii-1)+1:457*100*ii));
[vv,ll]=max(uu.o);
ui=reshape(ll-1,457,100);
un(:,100*(ii-1)+1:100*ii)=ui(:,1:100);

end
disp(['Part 10 Start']);
uu = cnnff(cnn, test_x(:,:,900*457+1:977*457));
[vv,ll2]=max(uu.o);
ui2=reshape(ll2-1,457,77);
un(:,901:977)=ui2(:,1:77);
disp(['Done...']);


%%%%%%%%%% Image the result %%%%%%%%%%%%
figure(1)
subplot(1,2,1)
imagesc(image_input)
title({['Input Image']},'fontsize',22,'color','black')
set(gca,'fontsize',22)
xlabel('Pixels','fontsize',22)
ylabel('Pixels','fontsize',22)
subplot(1,2,2)
imagesc(un,[0 nclass-1])
title({['CNN Labels (3 classes)']},'fontsize',22,'color','black')
set(gca,'fontsize',22)
xlabel('Pixels','fontsize',22)
ylabel('Pixels','fontsize',22)
set(gcf,'unit','centimeters','position',[12 5 30 13])
colormap(gca,parula(nclass))
cb=colorbar;
set(cb,'ytick',(0:nclass-1)*(nclass-1)/nclass+(nclass-1)/(2*nclass),'yticklabel',class_names)
%%%%%%%%%%Plot the error curve  %%%%%%%%%%%
figure(2)
plot(cnn.rL);
title({['Misfit Curve (3 classes)']},'fontsize',22,'color','black')
set(gca,'fontsize',22)
xlabel('Iteration','fontsize',22)
ylabel('Misfit','fontsize',22)

%%%%%%%%%% Picked training points %%%%%%%%%%%%
figure(3)
imagesc(image_input); hold on
mk={'go','co','ro'};
for ic=1:nclass
    idx=(ic-1)*ntrain+1:ic*ntrain;
    plot(train.trainx(idx),train.trainz(idx),mk{ic},'markersize',4,'linewidth',1.5);
end
hold off
legend(class_names,'location','northeastoutside')
title({['Training Set (',num2str(nclass*ntrain),' points)']},'fontsize',22,'color','black')
set(gca,'fontsize',22)



