clear
clc
close all

%%%%%%%%%%%%Auto Semblance Spectrum Picking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program use K-means methods to automatically pick the velocity spectrum. 
% Writen by Yuqing Chen on 2018.11.1
% E-mail: yuqing.chen@kaust.edu.sa
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%(1) Load a the velocity spectrum
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nt=3521; n_vel=381;                    % The size of velocity spectrum
fp=fopen('./vel_spec.bin775','rb');    % Read the velocity spectrum file
vel_spec=fread(fp,[nt,n_vel],'float');
fclose(fp);
vel_spec=medfilt2(vel_spec,[3,3]);     % Process the velocity spectrum

%(2) Thresholding the velocity spectrum
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thres_value = prctile(prctile(vel_spec,90),90);  % Set the threshold value
vel_spec2=(vel_spec>thres_value);                % Filter out small value

%(3) Build the training set. Only the points which their value larger than
%0 will be participate in K-means computation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_point=sum(sum(vel_spec>thres_value));   % Count the number of points larger than "thres_value"
train_set=zeros(n_point,2); % Each example in the training set has two features,
                            % First feature : horizental location
                            % Second feature : vertical location
i_point=1;                  % Build the training set 
for ix=1:n_vel
    for iz=1:nt
        if(vel_spec2(iz,ix)>0)
           train_set(i_point,1)=ix;
           train_set(i_point,2)=iz;
           i_point=i_point+1;
        end
    end
end

%(4) Training...
%%%%%%%%%%%%%%%%%

%(4.1) Initialize
%%%%%%%%%%%%%%%%%
% num_cluster=15;   % padrao
% num_cluster=6;    % otimo
num_cluster=9;      % um pouco acima do otimo
dx=floor(n_vel/(num_cluster+1));   % Set the initial location of each cluster centroid
dz=floor(nt/(num_cluster+1));
central=zeros(num_cluster,2);      % Matrix used to saved each cluster central

for ik=1:num_cluster               % Initialize each cluster central
    central(ik,1)=dx*ik;
    central(ik,2)=dz*ik;
end

maxiter=15;                        % Set the iteration number

%(4.2) Interation start...
%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter=1:maxiter
    % Plot each cluster central on the velocity spectrum
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    imagesc(vel_spec)
    hold on
    plot(central(:,1),central(:,2),'r*');pause(1);hold off
    
    % "num_cluster" clusters are created by associating every points with
    %  the nearest distance
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    res=0.0;
    Y=zeros(n_point,1);
    for ip=1:n_point         % Loop over each cluster central
        
        % Compute the distance of each points to the first cluster central
        dist1=(train_set(ip,1)-central(1,1))^2+(train_set(ip,2)-central(1,2))^2;
        Y(ip)=1;
        
        % Compute the distance of each points to the each cluster central
        for ik=2:num_cluster  % Loop over each cluster central    
            dist_temp=(train_set(ip,1)-central(ik,1))^2+(train_set(ip,2)-central(ik,2))^2;
            if(dist1>dist_temp)
               dist1=dist_temp;
               Y(ip)=ik;
            end
        end
        res=res+dist1;        % Compute residual
    end 
    
    fprintf('iter= %d res = %f\n',iter,res)
  
    % Compute the new central of each cluster
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    num_K=zeros(num_cluster,1);
    central_temp=zeros(size(central));
    for ip=1:n_point
        central_temp(Y(ip),1)=central_temp(Y(ip),1)+train_set(ip,1);
        central_temp(Y(ip),2)=central_temp(Y(ip),2)+train_set(ip,2);
        num_K(Y(ip))=num_K(Y(ip))+1;
    end
    
    % "num_K" saves the number of points in each cluster
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ik=1:num_cluster
        if(num_K(ik)==0) 
           num_K(ik)=1; 
        end
    end
    
    % Compute the average
    %%%%%%%%%%%%%%%%%%%%%
    for ik=1:num_cluster
        central_temp(ik,1)=central_temp(ik,1)/num_K(ik);
        central_temp(ik,2)=central_temp(ik,2)/num_K(ik);
    end
    
    central=central_temp;
   
end

%%%%%Plot Figure
xx=1400:10:5200;
zz=(1:nt)*0.001;
figure;subplot(1,2,1);imagesc(xx,zz,vel_spec);title('Velocity Spectrum')
xlabel('Velocity(m/s)')
ylabel('Time (s)')
subplot(1,2,2);imagesc(xx,zz,vel_spec);hold on;plot(1400+central(:,1)*10,central(:,2)*0.001,'k','linewidth',2)
hold on;plot(1400+central(:,1)*10,central(:,2)*0.001,'r*');title('Auto Pick Result')
xlabel('Velocity(m/s)')
ylabel('Time (s)')





