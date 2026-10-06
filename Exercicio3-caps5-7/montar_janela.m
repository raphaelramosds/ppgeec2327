function varargout = montar_janela(varargin)
% MONTAR_JANELA Unified window extractor/assembler for training data
%   Two usages are supported (single-file replacement for extrair_janela and montar_janela):
%   1) janela = montar_janela(mig, cx, cz, winx, winz)
%        - Extracts a winx-by-winz window centered at (cx,cz) from mig and
%          returns it as a 1-by-(winx*winz) row vector (flattened column-major).
%   2) X = montar_janela(X, iz, kk, xpt, zpt, mig, winx, winz)
%        - Extracts the window centered at (xpt(kk), zpt(kk)) and writes the
%          flattened window into row iz of matrix X, returning the updated X.
%
%   Behavior details (both usages):
%     - If the window would go out of image bounds, it is shifted inward to
%       keep the full window inside the image (no padding).
%     - Works with even/odd window sizes; the center is computed by rounding cx/cz.
%     - Throws an error if winx>nx or winz>nz (window larger than image).
%
%   Examples:
%     w = montar_janela(mig, 10, 20, 5, 5);           % returns 1x25 vector
%     X = montar_janela(X, 1, k, xpts, zpts, mig,5,5);% fills row 1 of X

if nargin==5
    % Extract and return a flattened window: (mig, cx, cz, winx, winz)
    mig = varargin{1}; cx = varargin{2}; cz = varargin{3}; winx = varargin{4}; winz = varargin{5};
    [nz, nx] = size(mig);
    if winx > nx || winz > nz
        error('Window size (%d,%d) is larger than image size (%d,%d).', winx, winz, nx, nz);
    end
    cx = round(cx);
    cz = round(cz);
    left = floor((winx - 1) / 2);
    right = winx - left - 1;
    up = floor((winz - 1) / 2);
    down = winz - up - 1;
    c1 = cx - left; c2 = cx + right;
    r1 = cz - up;   r2 = cz + down;
    if c1 < 1
        shift = 1 - c1; c1 = 1; c2 = min(nx, c2 + shift); end
    if c2 > nx
        shift = c2 - nx; c2 = nx; c1 = max(1, c1 - shift); end
    if r1 < 1
        shift = 1 - r1; r1 = 1; r2 = min(nz, r2 + shift); end
    if r2 > nz
        shift = r2 - nz; r2 = nz; r1 = max(1, r1 - shift); end
    winslice = mig(r1:r2, c1:c2);
    if ~isequal(size(winslice), [winz, winx])
        error('Invalid extracted window size %dx%d, expected %dx%d.', size(winslice,2), size(winslice,1), winx, winz);
    end
    janela = reshape(winslice, 1, []);
    varargout{1} = janela;

elseif nargin==8
    % Fill row in X: (X, iz, kk, xpt, zpt, mig, winx, winz)
    X = varargin{1}; iz = varargin{2}; kk = varargin{3}; xpt = varargin{4}; zpt = varargin{5}; mig = varargin{6}; winx = varargin{7}; winz = varargin{8};
    % Use the 5-arg extraction mode to get the flattened window
    janela = montar_janela(mig, xpt(kk), zpt(kk), winx, winz);
    X(iz,:) = janela;
    varargout{1} = X;
else
    error('montar_janela: invalid number of arguments. See help montar_janela for usage.');
end