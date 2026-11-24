clear
clc
close all

%%%%%%%%%%%% Elbow method for K-means velocity picking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Runs the same K-means used in K_mean_cluster.m for several values of K
% and plots the final residual "res" (sum of squared distances of each
% point to its centroid) as a function of K.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%(1) Load the velocity spectrum
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nt=3521; n_vel=381;
fp=fopen('./vel_spec.bin775','rb');
vel_spec=fread(fp,[nt,n_vel],'float');
fclose(fp);
vel_spec=medfilt2(vel_spec,[3,3]);

%(2) Thresholding (same as original program)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
thres_value = prctile(prctile(vel_spec,90),90);
vel_spec2=(vel_spec>thres_value);

%(3) Training set: [velocity index, time index] of the points above threshold
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[iz_list, ix_list] = find(vel_spec2);      % same ordering as the original loops
train_set = [ix_list, iz_list];
n_point   = size(train_set,1);

%(4) Loop over K
%%%%%%%%%%%%%%%%%
K_list  = 2:30;                 % Values of K to test
maxiter = 15;                   % Same number of iterations as the original
res_K   = zeros(size(K_list));

for iK = 1:numel(K_list)
    num_cluster = K_list(iK);

    % Initialization (same diagonal scheme as the original program)
    dx = floor(n_vel/(num_cluster+1));
    dz = floor(nt/(num_cluster+1));
    central = [dx*(1:num_cluster)', dz*(1:num_cluster)'];

    for iter = 1:maxiter
        % Assignment step: squared distance of every point to every centroid
        D = (train_set(:,1) - central(:,1)').^2 + ...
            (train_set(:,2) - central(:,2)').^2;     % n_point x num_cluster
        [dmin, Y] = min(D, [], 2);

        res = sum(dmin);    % Residual as in K_mean_cluster.m (res=res+dist1)

        % Update step: mean of the points in each cluster
        num_K = accumarray(Y, 1, [num_cluster 1]);
        sum_x = accumarray(Y, train_set(:,1), [num_cluster 1]);
        sum_z = accumarray(Y, train_set(:,2), [num_cluster 1]);
        num_K(num_K==0) = 1;                         % avoid division by zero
        central = [sum_x./num_K, sum_z./num_K];
    end

    res_K(iK) = res;
    fprintf('K = %2d   res = %e\n', num_cluster, res);
end

%(5) Plot
%%%%%%%%%%
figure;
plot(K_list, res_K, 'b-o', 'linewidth', 1.5);
grid on
xlabel('Number of clusters (K)')
ylabel('Misfit')
title('Elbow method')
