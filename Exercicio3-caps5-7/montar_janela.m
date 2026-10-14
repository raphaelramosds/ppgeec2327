function X = montar_janela(X, iz, kk, xpt, zpt, mig, winx, winz)
% MONTAR_JANELA Extrai uma janela e armazena os valores em uma linha da
% matriz de treinamento.
%   X = MONTAR_JANELA(X, iz, kk, xpt, zpt, mig, winx, winz)
%   extrai a janela centralizada em (xpt(kk), zpt(kk)) e grava seus valores
%   achatados na linha iz de X.
%
%   Se a janela ultrapassar os limites da imagem, ela sera deslocada para
%   dentro da imagem, mantendo a janela completa. Nenhum preenchimento
%   (padding) e utilizado.

[nz, nx] = size(mig);
if winx > nx || winz > nz
    error('Window size (%d,%d) is larger than image size (%d,%d).', winx, winz, nx, nz);
end

cx = round(xpt(kk));
cz = round(zpt(kk));
left = floor((winx - 1) / 2);
right = winx - left - 1;
up = floor((winz - 1) / 2);
down = winz - up - 1;

c1 = cx - left;
c2 = cx + right;
r1 = cz - up;
r2 = cz + down;

if c1 < 1
    shift = 1 - c1;
    c1 = 1;
    c2 = min(nx, c2 + shift);
end
if c2 > nx
    shift = c2 - nx;
    c2 = nx;
    c1 = max(1, c1 - shift);
end
if r1 < 1
    shift = 1 - r1;
    r1 = 1;
    r2 = min(nz, r2 + shift);
end
if r2 > nz
    shift = r2 - nz;
    r2 = nz;
    r1 = max(1, r1 - shift);
end

winslice = mig(r1:r2, c1:c2);
if ~isequal(size(winslice), [winz, winx])
    error('Invalid extracted window size %dx%d, expected %dx%d.', ...
        size(winslice, 2), size(winslice, 1), winx, winz);
end

X(iz,:) = reshape(winslice, 1, []);
end