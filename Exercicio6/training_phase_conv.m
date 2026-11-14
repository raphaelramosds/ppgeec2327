%% Comparacao: filtros do CSC x filtros da 1a camada de uma CNN
% Questao 2 do Exercicio 6.
%  (1) Plota os filtros d aprendidos pelo CSC (filters.mat, gerado por training_phase.m)
%  (2) Plota os filtros conv1 da AlexNet (se o support package estiver instalado)
%  (3) Treina uma CNN rasa no MESMO sismograma usado no CSC, com o MESMO
%      tamanho/numero de filtros, de forma supervisionada (ruidoso -> limpo),
%      e plota os filtros da 1a camada convolucional.

clear;
close all;
rng(0); % reprodutibilidade

outdir = fullfile('imagens', 'q2');
if ~exist(outdir, 'dir'), mkdir(outdir); end

%% (1) Filtros do CSC
filters = load('filters.mat');
d = filters.d;                 % [n1 x n2 x K]
filter_size = size(d);
K = filter_size(3);

fig1 = figure('Name', 'Filtros CSC');
imagesc(filter_mosaic(d)), colormap gray,
axis image, colorbar, title('Final filter estimate (CSC)');
saveas(fig1, fullfile(outdir, 'filtros_csc.jpg'));

%% (2) Filtros conv1 da AlexNet (opcional)
try
    net_alex = alexnet;
    w = net_alex.Layers(2).Weights;          % 11 x 11 x 3 x 96
    w = (w - min(w(:))) / (max(w(:)) - min(w(:)));
    fig2 = figure('Name', 'AlexNet conv1');
    montage(w);                              % mosaico 11x11 RGB
    title('AlexNet - filtros da 1a camada (96 x 11x11x3)');
    saveas(fig2, fullfile(outdir, 'filtros_alexnet_conv1.jpg'));
catch err
    fprintf('AlexNet indisponivel (%s)\nPulando a etapa (2).\n\n', err.message);
end

%% (3) CNN rasa treinada no mesmo dado sismico
% Mesmo pre-processamento de training_phase.m
load b.mat
b1 = b(1:100, :, 1);
x = local_contrast_normalization(b1, 21);
x = x - mean(x(:));

% Pares de treinamento (patch ruidoso -> patch limpo), mesmo nivel de
% ruido usado em denoise.m
patch = [41 41];
n_patches = 3000;
noise_level = 0.3;
[n1, n2] = size(x);
X = zeros(patch(1), patch(2), 1, n_patches, 'single');
Y = zeros(patch(1), patch(2), 1, n_patches, 'single');
for p = 1:n_patches
    i0 = randi(n1 - patch(1) + 1);
    j0 = randi(n2 - patch(2) + 1);
    clean = x(i0:i0+patch(1)-1, j0:j0+patch(2)-1);
    if rand > 0.5, clean = fliplr(clean); end   % aumento de dados simples
    Y(:, :, 1, p) = clean;
    X(:, :, 1, p) = clean + noise_level * randn(patch);
end

% Arquitetura: conv(KxN1xN2) -> ReLU -> conv 1x1 (combina mapas) -> regressao
% padding [p_vert p_horiz] mantem o tamanho da saida igual ao da entrada
pad = [(filter_size(1)-1)/2, (filter_size(2)-1)/2];
layers = [
    imageInputLayer([patch 1], 'Normalization', 'none')
    convolution2dLayer(filter_size(1:2), K, 'Padding', pad, 'Name', 'conv1')
    reluLayer
    convolution2dLayer(1, 1, 'Name', 'conv_out')
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'cpu', ...
    'Verbose', false, ...
    'OutputFcn', @(info) print_progress(info, 50), ...
    'Plots', 'none');

tic();
net = trainNetwork(X, Y, layers, options);
fprintf('Tempo de treino da CNN: %.1f s\n', toc());

% Filtros da 1a camada da CNN
w_cnn = net.Layers(2).Weights;   % n1 x n2 x 1 x K
d_cnn = double(squeeze(w_cnn));  % n1 x n2 x K (mesmo formato de d)
fig3 = figure('Name', 'Filtros CNN');
imagesc(filter_mosaic(d_cnn)), colormap gray,
axis image, colorbar, title('Final filter estimate (CNN conv1)');
saveas(fig3, fullfile(outdir, 'filtros_cnn_conv1.jpg'));

%% Comparacao lado a lado e metricas simples
fig4 = figure('Name', 'CSC x CNN');
subplot(1, 2, 1);
imagesc(filter_mosaic(d)), colormap gray,
axis image, colorbar, title('CSC');
subplot(1, 2, 2);
imagesc(filter_mosaic(d_cnn)), colormap gray,
axis image, colorbar, title('CNN conv1');
saveas(fig4, fullfile(outdir, 'csc_vs_cnn.jpg'));

% Esparsidade das representacoes: fracao de coeficientes ~ 0
z_csc = filters.z;
frac_csc = mean(abs(z_csc(:)) < 1e-3);
act = activations(net, X(:, :, :, 1:200), 'conv1', ...
    'ExecutionEnvironment', 'cpu');          % evita GPU tambem na inferencia
act = max(act, 0);                          % saida apos a ReLU
frac_cnn = mean(act(:) < 1e-3);
fprintf('Fracao de coeficientes ~0 no CSC (z):          %.2f\n', frac_csc);
fprintf('Fracao de ativacoes ~0 na CNN (apos ReLU):     %.2f\n', frac_cnn);

% Normas dos filtros (CSC tem restricao ||d_k||_2 <= 1; CNN nao tem)
for k = 1:K
    fprintf('Filtro %d: ||d_csc||_2 = %.3f   ||w_cnn||_2 = %.3f\n', k, ...
        norm(reshape(d(:, :, k), [], 1)), norm(reshape(w_cnn(:, :, 1, k), [], 1)));
end

save(fullfile(outdir, 'cnn_conv1.mat'), 'net', 'w_cnn');

%% Funcoes locais
function stop = print_progress(info, freq)
% Imprime o progresso do treino a cada freq iteracoes.
% MSE = TrainingRMSE^2 (erro quadratico medio por pixel do mini-batch)
stop = false;
if strcmp(info.State, 'iteration') && ...
        (info.Iteration == 1 || mod(info.Iteration, freq) == 0)
    fprintf('Epoch %d, Iter %d, MSE %5.5g \n', ...
        info.Epoch, info.Iteration, info.TrainingRMSE^2);
end
end

function d_disp = filter_mosaic(d)
% Monta os filtros d (n1 x n2 x K) em um mosaico sqr_k x sqr_k com borda
% de 1 pixel, como na secao "Show filters result" de training_phase.m
pd = 1;
fs = [size(d, 1), size(d, 2)];
sqr_k = ceil(sqrt(size(d, 3)));
d_disp = zeros(sqr_k * [fs(1) + pd, fs(2) + pd] + [pd, pd]);
for j = 0:size(d, 3) - 1
    d_disp(floor(j/sqr_k) * (fs(1) + pd) + pd + (1:fs(1)), ...
           mod(j, sqr_k) * (fs(2) + pd) + pd + (1:fs(2))) = d(:, :, j + 1);
end
end
