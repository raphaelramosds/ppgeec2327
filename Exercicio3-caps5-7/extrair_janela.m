function janela = extrair_janela(mig, cx, cz, winx, winz)
% extrair_janela Deprecated wrapper for montar_janela
%   This function is kept for backward compatibility. Use montar_janela(mig,cx,cz,winx,winz)
%   instead. It simply forwards the call to montar_janela.

janela = montar_janela(mig, cx, cz, winx, winz);

[nz, nx] = size(mig);

if winx > nx || winz > nz
    error('Window size (%d,%d) is larger than image size (%d,%d).', winx, winz, nx, nz);
end

cx = round(cx); % x = coluna
cz = round(cz); % z = linha

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

winslice = mig(r1:r2, c1:c2); % ordem MATLAB: (linha, coluna) = (z, x)
if ~isequal(size(winslice), [winz, winx])
    error('Invalid extracted window size %dx%d, expected %dx%d.', size(winslice,2), size(winslice,1), winx, winz);
end

janela = reshape(winslice, 1, []);
end
