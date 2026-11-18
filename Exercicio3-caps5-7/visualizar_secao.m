clear
clc
close all

% Visualizacao da secao sismica migrada original e de seus atributos
% (amplitude, coerencia e fase/angulo local), sem nenhum processamento.

load 'mig_and_features.mat'   % mig, cohe, angl, engy

figure('Name', 'Secao sismica e atributos');

subplot(2,2,1)
imagesc(Normalize(mig), [-0.06 0.06]); colormap(gca, gray); colorbar
title('Secao migrada original'); xlabel('x'); ylabel('z')

subplot(2,2,2)
imagesc(engy); colormap(gca, jet); colorbar
title('Amplitude'); xlabel('x'); ylabel('z')

subplot(2,2,3)
imagesc(cohe); colormap(gca, jet); colorbar
title('Coerencia'); xlabel('x'); ylabel('z')

subplot(2,2,4)
imagesc(angl); colormap(gca, jet); colorbar
title('Fase (angulo local)'); xlabel('x'); ylabel('z')
