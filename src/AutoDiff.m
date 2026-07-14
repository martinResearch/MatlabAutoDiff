% License FreeBSD:
%
% Copyright (c) 2016  Martin de La Gorce, with extensions
% by Chris Noble C.2025, and Tony Roberts (AJR) June 2026.
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or
% without modification, are permitted provided that the
% following conditions are met:
%
% 1. Redistributions of source code must retain the above
% copyright notice, this list of conditions and the
% following disclaimer.
% 2. Redistributions in binary form must reproduce the above
% copyright notice, this list of conditions and the
% following disclaimer in the documentation and/or other
% materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
% CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
% WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
% PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
% OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
% INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
% GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
% BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
% LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
% OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
%
% The views and conclusions contained in the software and
% documentation are those of the authors and should not be
% interpreted as representing official policies, either
% expressed or implied, of the FreeBSD Project.
% 
% Sometimes info obtained by command such as "help AutoDiff.svd"


classdef AutoDiff

    %
    %    This class implement a forward automatic differentation
    %    method based on operator overloading. This class allows
    %    precise and efficient computation of function Jacobians
    %    by calling AutoDiffJacobianAutoDiff
    %
    %   In contrast with most AD matlab tools
    %    - Derivatives are represented as sparse matrices
    %    - N dimensional array are supported
    %
    %   The speed could be improved by representing jacobian
    %   matrices by their transposed matrix, due to the way
    %   matlab store sparse matrices
    %


    properties
        values
        derivatives
    end

    methods

        function x = AutoDiff(values, derivatives)
            if isstruct(values)
                x.values = values.values;
                x.derivatives = values.derivatives;
            else
                x.values = values;
                if nargin == 1
                    x.derivatives = speye(numel(values));
                else
                    if isa(derivatives, 'AutoDiff')
                        x.derivatives = sparse(numel(values), size(derivatives.derivatives, 2));
                    else
                        x.derivatives = derivatives;
                    end
                end
            end
        end

        function Jac = getderivs(x)
            Jac = x.derivatives;
        end

        function val = getvalue(x)
            val = x.values;
        end

        function x = setdervis(x, derivatives)
            x.derivatives = derivatives;
        end

        function x = double(~)
            error(['Conversion to double from AutoDiff is not possible.\n', ...
                'This might be due to preallocation of the array on the left side of the assignement.\n', ...
                'Considere modifying the code by either\n', ...
                ' - vectorizing your code to void preallocation or that array\n', ...
                ' - create the prealocated array as AutoDiff using zeros(m,n,''like'', x) or ones(m,n,''like'', x) with x the differentiated input', ...
                ' - create the prealocated array as AutoDiff when needed with the right size derivatives using autodiff_identity\n', ...
                ' - create a cell array of the elements and then concatenate them\n', ...
                'see troubleshoot example 1 in autodiff_troubleshoot.m for a more detailed examples and solutions\n', ...
                'Note that vectorizing your code is likely to avoid the preallocation is likely to lead to faster execution']);
        end

        function y = isinf(x)
        % isinf() for AutoDiff x is true for infinite values
            y = isinf(x.values);
        end

        function x = sinh(x)
            x.derivatives = AutoDiff.spdiag(cosh(x.values)) * x.derivatives;
            x.values = sinh(x.values);
        end
        
        function x = cosh(x)
            x.derivatives = AutoDiff.spdiag(sinh(x.values)) * x.derivatives;
            x.values = cosh(x.values);
        end

        function x = asinh(x)
            x.derivatives = AutoDiff.spdiag(1./sqrt(1+(x.values).^2)) * x.derivatives;
            x.values = asinh(x.values);
        end

        function x = abs(x)
            x.derivatives = AutoDiff.spdiag(sign(x.values)) * x.derivatives;
            x.values = abs(x.values);
        end

        function x = acosh(x)
            x.derivatives = AutoDiff.spdiag(1./sqrt((x.values).^2-1)) * x.derivatives;
            x.values = acosh(x.values);
        end

        function x = atanh(x)
            x.derivatives = AutoDiff.spdiag(1./(1-(x.values).^2)) * x.derivatives;
            x.values = atanh(x.values);
        end

        function x = sqrt(x)
            x.values = sqrt(x.values);
            x.derivatives = AutoDiff.spdiag(0.5./x.values) * x.derivatives;
        end

        function x = cos(x)
            x.derivatives = AutoDiff.spdiag(-sin(x.values)) * x.derivatives;
            x.values = cos(x.values);
        end

        function x = sin(x)
            x.derivatives = AutoDiff.spdiag(cos(x.values)) * x.derivatives;
            x.values = sin(x.values);
        end

        function x = tan(x)
            tmp = 1 ./ cos(x.values).^2;
            x.derivatives = AutoDiff.spdiag(tmp) * x.derivatives;
            x.values = tan(x.values);
        end

        function x = acos(x)
            x.derivatives = AutoDiff.spdiag(-1./sqrt(1 - x.values.^2)) * x.derivatives;
            x.values = acos(x.values);
        end

        function x = asin(x)
            x.derivatives = AutoDiff.spdiag(1./sqrt(1 - x.values.^2)) * x.derivatives;
            x.values = asin(x.values);
        end

        function y = ceil(x)
            y = ceil(x.values);
        end

        function y = floor(x)
            y = floor(x.values);
        end
        
        function x = real(x)
            x.derivatives = real(x.derivatives);%added ; AJR, 3 Jun 2026
            x.values = real(x.values);
        end

        function x = imag(x)
            x.derivatives = imag(x.derivatives);%added ; AJR, 3 Jun 2026
            x.values = imag(x.values);
        end
        
        function x = atan(x)
            x.derivatives = AutoDiff.spdiag(1./(1 + x.values.^2)) * x.derivatives;
            x.values = atan(x.values);
        end

        function x = erf(x)
        % erf(x) for AutoDiff x is Error function and derivatives
            tmp = 2*exp(-1*x.values.^2)/sqrt(pi);
            x.derivatives = AutoDiff.spdiag(tmp) * x.derivatives;
            x.values = erf(x.values);
        end

        function x = erfc(x)
        % erfc(x) for AutoDiff x is Complementary Error function and
        % derivatives.
            tmp = 2*exp(-1*x.values.^2)/sqrt(pi);
            x.derivatives = -1*AutoDiff.spdiag(tmp) * x.derivatives;
            x.values = erfc(x.values);
        end

        function x = gamma(x)
        % gamma(x) for AutoDiff x is Gamma function and derivatives
            tmp = gamma(x.values).*psi(x.values);
            x.values = gamma(x.values);
            x.derivatives = AutoDiff.spdiag(tmp) * x.derivatives;
        end

        function x = exp(x)
            x.values = exp(x.values);
            x.derivatives = AutoDiff.spdiag(x.values) * x.derivatives;
        end

        function x = log(x)
            tmp = 1 ./ x.values;
            x.derivatives = AutoDiff.spdiag(tmp) * x.derivatives;
            x.values = log(x.values);
        end

        function x = tanh(x)
            x.derivatives = AutoDiff.spdiag(1./(cosh(x.values).^2)) * x.derivatives;
            x.values = tanh(x.values);
        end

        function x = conj(x)
            x.values = conj(x.values);
            x.derivatives = conj(x.derivatives);
        end

        function y = fft(x,varargin)
            % y=fft(x,...) for AutoDiff x gives a 1D Fourier Transform
            % of x and its derivatives.  Optional arguments are passed
            % to the usual fft.  Cater for sparse derivatives as in many
            % problems, although the fft-dirn may be dense, a lot of the
            % variables may have no influence (e.g., Eqn-Free Patch
            % scheme), that is, the 'columns' of derivatives would be
            % sparse.   
            %
            % Original C.2025 by Chris Noble, github.com/noblec04/MatlabGP
            % Vectorized by AJR 7 Jun 2026.
            y.values = fft(x.values,varargin{:});
            nz = size(x.derivatives,2);
            nx = numel(x.values);
            inz = find(~all(x.derivatives==0)); % set of non-zero columns
            ninz = numel(inz);  % number of non-zero columns
            y.derivatives = sparse(nx,nz);
            if ninz==0, y = AutoDiff(y); return, end
            if 0 % original version by CN, modified
                for i = inz
                dxi = reshape(full(x.derivatives(:,i)),size(x.values));
                dyi = fft(dxi,varargin{:});
                y.derivatives(:,i) = dyi(:);
                end%for
            else % AJR, vectorized version may be roughly 10x faster
                Dxi = reshape(full(x.derivatives(:,inz)),[size(x.values) ninz]);
                Dxi = fft(Dxi,varargin{:});
                y.derivatives(:,inz) = reshape(Dxi,nx,ninz);
            end%if 01-option
            y = AutoDiff(y);
        end

        function y = ifft(x,varargin)
            % y=ifft(x,...) for AutoDiff x gives a 1D inverse Fourier
            % Transform of x and its derivatives.  Optional arguments
            % are passed to the usual fft.  Cater for sparse
            % derivatives.   
            %
            % Original C.2025 by Chris Noble, github.com/noblec04/MatlabGP
            % Vectorized by AJR 7 Jun 2026.
            y.values = ifft(x.values,varargin{:});
            nz = size(x.derivatives,2);
            nx = numel(x.values);
            inz = find(~all(x.derivatives==0)); % set of non-zero columns
            ninz = numel(inz);  % number of non-zero columns
            y.derivatives = sparse(nx,nz);
            if ninz==0, y = AutoDiff(y); return, end
            if 0 % original version by CN, modified
                for i = inz
                dxi = reshape(full(x.derivatives(:,i)),size(x.values));
                dyi = ifft(dxi,varargin{:});
                y.derivatives(:,i) = dyi(:);
                end%for
            else % AJR, vectorized version may be roughly 10x faster
                Dxi = reshape(full(x.derivatives(:,inz)),[size(x.values) ninz]);
                Dxi = ifft(Dxi,varargin{:});
                y.derivatives(:,inz) = reshape(Dxi,nx,ninz);
            end%if
            y = AutoDiff(y);
        end
        
        function b = isreal(x)
            b = isreal(x.values);
        end


        function y = cat(dim, varargin)

            y.values = [];
            nbvarargin = nargin - 1;

            % get the number of derivatives
            for i = 1:nbvarargin
                x = varargin{i};
                if isa(x, 'AutoDiff')
                    nderivs = size(x.derivatives, 2);
                    break;
                end
            end


            for i = 1:nbvarargin
                x = varargin{i};
                if ~isa(x, 'AutoDiff')
                    if ~isempty(x)
                        y.values = cat(dim, y.values, x);
                    end
                else
                    y.values = cat(dim, y.values, x.values);
                    if nderivs ~= size(x.derivatives, 2)
                        error('AutoDiff:NonUniformDerivatesNumber', 'The number of derivatives is not uniform');
                    end
                end
            end

            nvars = numel(y.values);
            y.derivatives = sparse(nvars, nderivs);
            sy = size(y.values);
            sy(numel(sy)+1:dim+1) = 1;
            nr = 1;

            for i = 1:nbvarargin
                x = varargin{i};                
                if isa(x, 'AutoDiff')
                    sx = size(x.values);
                    sx(numel(sx)+1:dim+1) = 1;
                    [k, l] = meshgrid(1:prod(sx(dim + 1:end)), 1:prod(sx(1:dim)));
                    idx = l(:) + nr - 1 + (k(:) - 1) * prod(sy(1:dim));
                    M = sparse(idx, 1:numel(idx), ones(1, numel(idx)), nvars, numel(idx));
                    y.derivatives = y.derivatives + M * x.derivatives;
                else
                    sx = size(x);
                    sx(numel(sx)+1:dim+1) = 1;
                end
                nr = nr + prod(sx(1:dim));
            end

            y = AutoDiff(y.values, y.derivatives);
        end

        function x = repmat(x, varargin)
            r = repmat(reshape((1:numel(x.values)), size(x.values)), varargin{:});
            x.values = x.values(r);
            x.derivatives = sparse(1:numel(r), r(:), ones(numel(r),1)) * x.derivatives;
        end

        function x = ctranspose(x)
            x = transpose(x);
            if ~isreal(x)
                x = conj(x);
            end
        end

        function D = spdiags(B, d, m, n)
            if isvector(B)
                N = numel(B);
                i = 1:N;
                D.values = spdiags(B.values, d, m, n);
                id = (N + 1) * i - N;
                D.derivatives = sparse(id, i, ones(1, N)) * B.derivatives;
                D = AutoDiff(D.values, D.derivatives);
            else
                error('not yet coded')
            end
        end

        function D = diag(M)


            if isvector(M)
                N = numel(M);
                i = 1:N;
                D.values = diag(M.values);
                id = (N + 1) * i - N;
                D.derivatives = sparse(id, i, ones(1, N)) * M.derivatives;
            else
                N = min(size(M, 1));
                i = 1:N;
                D.values = diag(M.values);
                id = (N + 1) * i - N;
                D.derivatives = sparse(i, id, ones(1, N)) * M.derivatives;
            end

            D = AutoDiff(D.values, D.derivatives);
        end

        function x = diff(x, n, dim)
            if nargin<3, dim=min(find(size(x)>1)); end %AJR, 31 May 2026
            if nargin<2, n=1;   end %AJR, 28 May 2026
            if n>1 %AJR, 28 May 2026, recursion probably inefficient
                x = diff(x,n-1,dim);
            end
            if issparse(x.values)
                warning('AutoDiff:Inefficient', 'this implementation is quite inefficent')
            end
            s = size(x.values);
          
            t = reshape(1:numel(x.values), [prod(s(1:dim-1)),s(dim), prod(s(dim+1:end))]);
           
            tsub1 = t(:, 2:end, :);
            tsub2 = t(:, 1:end-1, :);
            D = sparse(1:numel(tsub1), tsub1(:)', ones(1, numel(tsub1)), numel(tsub1), size(x.derivatives, 1)) - ...
                sparse(1:numel(tsub2), tsub2(:)', ones(1, numel(tsub2)), numel(tsub1), size(x.derivatives, 1));
            s(dim) = s(dim)-1;
            x.values = reshape(D*x.values(:), s);
            x.derivatives = D * x.derivatives;

        end

        function idx = end (x, k, n)
            if k == 1 && n == 1
                idx = length(x.values);
                return
            end
            idx = size(x.values, k);
        end

        function z = eq(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    z = x.values == y.values;
                else
                    z = x == y.values;
                end
            else
                z = x.values == y;
            end
        end

        function z = ne(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    z = x.values ~= y.values;
                else
                    z = x ~= y.values;
                end
            else
                z = x.values ~= y;
            end
        end

        function z = sign(x)
            z = sign(x.values);
        end

        function x = subsindex(x)
            x = x.values;
        end

        function z = ge(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    z = x.values >= y.values;
                else
                    z = x >= y.values;
                end
            else
                z = x.values >= y;
            end
        end

        function z = gt(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    z = x.values > y.values;
                else
                    z = x > y.values;
                end
            else
                z = x.values > y;
            end
        end


        function z = le(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    z = x.values <= y.values;
                else
                    z = x <= y.values;
                end
            else
                z = x.values <= y;
            end
        end


        function z = lt(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    z = x.values < y.values;
                else
                    z = x < y.values;
                end
            else
                z = x.values < y;
            end
        end

        function y = isnan(x)
            y = isnan(x.values);
        end


        function mylength = length(x)
            mylength = length(x.values);
        end


        function [m, id] = max(C, B)
            if nargin == 1
                if isvector(C.values)
                    [~, id] = max(C.values);
                    m.values = C.values(id);
                    m.derivatives = C.derivatives(id, :);       
                else
                    [v, id] = max(C.values);
                    id2 = id(:)' + (0:numel(id) - 1) * size(C.values, 1);
                    m.values = v;
                    tmp = sparse(1:numel(id2), id2(:)', ones(1, numel(id2)), numel(id2), size(C.derivatives, 1));
                    m.derivatives = tmp * C.derivatives;
                end
            elseif nargin == 2                
                if isa(C, 'AutoDiff')
                    if isa(B, 'AutoDiff')
                        m.values = max(C.values, B.values);
                        b = C.values > B.values;
                        m.derivatives = AutoDiff.spdiag(b) * C.derivatives + AutoDiff.spdiag(~b) * B.derivatives;
                    else
                        m.values = max(C.values, B);
                        b = C.values > B;
                        m.derivatives = AutoDiff.spdiag(b) * C.derivatives;
                    end
                else
                    m.values = max(C, B.values);
                    b = B.values > C;
                    m.derivatives = AutoDiff.spdiag(b) * B.derivatives;
                end
            else
                error('not coded yet')
            end
            m = AutoDiff(m);
        end

        function [m, id] = min(C, B)
            if nargin == 1
                if isvector(C.values)
                    [~, id] = min(C.values);
                    m.values = C.values(id);
                    m.derivatives = C.derivatives(id, :);
                else
                    [v, id] = min(C.values);
                    id2 = id(:)' + (0:numel(id) - 1) * size(C.values, 1);
                    m.values = v;
                    tmp = sparse(1:numel(id2), id2(:)', ones(1, numel(id2)), numel(id2), size(C.derivatives, 1));
                    m.derivatives = tmp * C.derivatives;
                end
            elseif nargin == 2                
                if isa(C, 'AutoDiff')
                    if isa(B, 'AutoDiff')
                        m.values = min(C.values, B.values);
                        b = C.values < B.values;
                        m.derivatives = AutoDiff.spdiag(b) * C.derivatives + AutoDiff.spdiag(~b) * B.derivatives;
                    else
                        m.values = min(C.values, B);
                        b = C.values < B;
                        m.derivatives = AutoDiff.spdiag(b) * C.derivatives;
                    end
                else
                    m.values = min(C, B.values);
                    b = B.values < C;
                    m.derivatives = AutoDiff.spdiag(b) * B.derivatives;
                end
            else
                error('not coded yet')
            end
            m = AutoDiff(m);
        end


        function x = minus(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    x = repmat_as(x, y);
                    y = repmat_as(y, x);
                    x.values = x.values - y.values;
                    x.derivatives = x.derivatives - y.derivatives;
                else
                    x = repmat_as(x, y);
                    y = repmat_as(y, x);
                    y.values = x - y.values;
                    y.derivatives = -y.derivatives;
                    x = y;
                end
            else
                x = repmat_as(x, y);
                y = repmat_as(y, x);
                x.values = x.values - y;
            end
        end


        function x = mpower(x, n)
            if numel(x) == 1
                x = x.^n;
            else
                if n == 1
                    return
                elseif n > 1
                    x = mtimes(x^(n - 1), x);
                else
                    error('not coded yet')
                end
            end
        end

        function x = inv(x)
            x.values = inv(x.values);
            M1 = kron(speye(size(x.values, 2)), x.values);
            M2 = kron(x.values', speye(size(x.values, 1)));
            x.derivatives = -M2 * M1 * x.derivatives;
        end

        function x = pinv(x,tol)
        % pinv(x) for AutoDiff x is Moore-Penrose pseudoinverse of
        % x, and derivatives. By Chris Noble C.2025
            if nargin==1
                tol=0;
            end
            x.values = pinv(x.values,tol);
            M1 = kron(speye(size(x.values, 2)), x.values);
            M2 = kron(x.values', speye(size(x.values, 1)));
            x.derivatives = -M2 * M1 * x.derivatives;
        end

        function [y,flag] = chol(x,shape)
        % chol(x,shape) for AutoDiff x is Cholesky factorization of
        % x, and derivatives. By Chris Noble C.2025
            if nargin<2, shape='lower'; end
            [L,flag] = chol(x.values,shape);
            if flag~=0, y=L; return, end
            for i = 1:size(x.derivatives,2)
                A = L\reshape(x.derivatives(:,i),size(x))/L';
                n = size(A,1);
                A(1:(n+1):end) = 0.5*A(1:(n+1):end);
                U = tril(0*A+1);
                A(U~=1) = 0;
                A = L*A;
                y.derivatives(:,i) = A(:);
            end
            y.values = L;
            y = AutoDiff(y);
        end

        function z = mldivide(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    z.values = x.values \ y.values;
                    if size(y, 2) > 1
                        error('not yet implemented')
                    end
                    z.derivatives = x.values \ (y.derivatives - kron(z.values', speye(size(x, 1))) * x.derivatives);
                    z = AutoDiff(z);

                else
                    z.values = x \ y.values;
                    z.derivatives = kron(speye(size(y, 2)), x) \ y.derivatives;
                    % might be inefficent....
                    z = AutoDiff(z);
                end
            else
                z.values = x.values \ y;
                if size(y, 2) > 1
                    error('not yet implemented')
                end
                z.derivatives = -x.values \ (kron(z.values', speye(size(x, 1))) * x.derivatives);
                z = AutoDiff(z);
            end
        end


        function z = mtimes(x, y)
            if (numel(x) == 1) || (numel(y) == 1)
                z = x .* y;
                return;
            end
            if (~isa(x, 'AutoDiff')) && (size(y.values, 2) == 1)
                z.values = x * y.values;
                z.derivatives = sparse(x) * y.derivatives;
                z = AutoDiff(z.values, z.derivatives);
            else

                if isa(x, 'AutoDiff')
                    if isa(y, 'AutoDiff')
                        z.values = x.values * y.values;
                        Mx = kron(speye(size(y.values, 2)), x.values);
                        My = kron(y.values', speye(size(x.values, 1)));
                        z.derivatives = Mx * y.derivatives + My * x.derivatives;

                    else
                        z.values = x.values * y;
                        My = kron(y', speye(size(x, 1)));
                        z.derivatives = My * x.derivatives;

                    end
                else
                    z.values = x * y.values;
                    Mx = kron(speye(size(y, 2)), x);
                    z.derivatives = Mx * y.derivatives;
                end
                z = AutoDiff(z.values, z.derivatives);
            end
        end

        function z = pagemtimes(x, y)

            if (numel(x) == 1) || (numel(y) == 1)
                z = x .* y;
                return;
            end
            
            if isa(y, 'AutoDiff')
                size_y = size(y.values);
                y_values = y.values;
            else
                size_y = size(y);
                y_values = y;
            end
            if isa(x, 'AutoDiff')
                size_x = size(x.values);
                x_values = x.values;
            else
                size_x = size(x);
                x_values = x;
            end 
            if ndims(x)~=ndims(y)|| any(size_x(3:end)~=size_y(3:end))
                new_size_x= size_y;
                new_size_x(1)=size(x,1);
                new_size_x(2)=size(x,2);
                if isa(x, 'AutoDiff')     
                    x = repmat_to_size(x, new_size_x);
                    x_values = x.values;
                else
                    x = x.* ones(new_size_x);
                    x_values = x;
                end
                new_size_y= size_x;
                new_size_y(1)=size_y(1);
                new_size_y(2)=size_y(2);
                if isa(y, 'AutoDiff')
                    y = repmat_to_size(y,new_size_y);
                    y_values = y.values;
                else
                    y = y.* ones(new_size_x);
                    y_values = y;
                end               
                size_x = size(x);
                size_y = size(y);
            end
      
            if isa(x, 'AutoDiff')
                s=prod(size_y(3:end));
                [i,j,k,l]=ndgrid(1:size_x(1),1:size_x(2),1:size_y(2),1:s);                    
                i2 = i+(k-1)*size_x(1)+(l-1)*size_x(1)*size_y(2);
                j2 = i+(j-1)*size_x(1)+(l-1)*size_x(1)*size_x(2);
                v=repmat(reshape(y_values,1,size_y(1),size_y(2),prod(size_y(3:end))),size_x(1),1,1,1);
                My=sparse(i2(:),j2(:),v(:));
            end
            if isa(y, 'AutoDiff')
                s=prod(size_x(3:end));
                [i,j,k,l]=ndgrid(1:size_x(1),1:size_y(1),1:size_y(2),1:s);                    
                j2 = j+(k-1)*size_y(1)+(l-1)*size_y(1)*size_y(2);
                i2 = i+(k-1)*size_x(1)+(l-1)*size_x(1)*size_y(2);
                v=repmat(reshape(x_values,size_x(1),size_x(2),prod(size_x(3:end))),1,size_y(2),1);
                Mx=sparse(i2(:),j2(:),v(:)); 
            end
            if isa(x, 'AutoDiff')
                if isa(y, 'AutoDiff')
                    z.values = pagemtimes(x.values,y.values);                 
                    z.derivatives = Mx * y.derivatives + My * x.derivatives;
                else
                    z.values = pagemtimes(x.values,y);
                    z.derivatives = My * x.derivatives;
                end
            else
                z.values = pagemtimes(x,y.values);          
                z.derivatives = Mx * y.derivatives;
            end
            z = AutoDiff(z.values, z.derivatives);            
        end
        
        
        function z = mrdivide(x, y)
            if (numel(y) == 1)
                z = x ./ y;
                return;
            else
                error('not yet coded')
            end
        end


        function x = norm(x, p)
            if nargin == 1
                p = 2;
            end

            if isvector(x)
                x = sum(abs(x.^p)).^(1 / p);
            elseif ismatrix(x)
                [~, d, ~] = svd(x);
                x = max(d);
            else
                error('not sure what matlab does in this case');
            end
        end

        function [Uad, Sad, Vad] = svd(A)
            % Economy-size SVD A=USV' for AutoDiff matrix A, mxn real or
            % complex: gives mxk U, kxk S, nxk V and its derivatives,
            % for rank k=number of non-zero singular values
            % [k<=min(m,n)]. Assumes there are no coincident singular
            % values (as otherwise F divides by zero).  Only compute the
            % economy rank k decomposition because for (multiple)
            % zero-rows/columns of S the corresponding columns of U&V
            % are not unique, and so their derivative is meaningless.
            %
            % AJR 22/6/2026, much revised from Chris Noble's MatlabGP,
            % and see "Differentiating the SVD", James Townsend, 2016
            %
            % AutoDiffsvdRankThresh (global) sets threshold on singular
            % values for determining the rank of the SVD and its
            % derivatives: singular values such that s_i/s_1 < threshold
            % are zeroed, and rank reduced accordingly.  Reason? 
            % derivatives of U&V involve division by singular values so
            % are very sensitive to very small ones.
            global AutoDiffsvdRankThresh
            if ~( exist('AutoDiffsvdRankThresh') ...
                    && ~isempty(AutoDiffsvdRankThresh) )
                AutoDiffsvdRankThresh = 1e-10;  % guess useful default
            end%if exist
            [m,n] = size(A.values);
            [U,S,V] = svd(A.values);
            s = diag(S);
            k = max(find(s>AutoDiffsvdRankThresh*s(1))); 
            s = s(1:k);     % kx1 col.vector of singular values
            U = U(:,1:k);   % mxk as in T2016
            S = diag(s);    % kxk matrix
            V = V(:,1:k);   % nxk as in T2016
            % 'rotate' each column of U&V so largest-V is real-positive
            % Means computed U&V is almost-always unique.
            [~,iVM]=max(abs(V));
            c=nan(1,k);
            for j=1:k, c(j)=conj(V(iVM(j),j))/abs(V(iVM(j),j)); end
            U=U.*c;  V=V.*c;
            
            N = size(A.derivatives,2);
            dU = sparse(m*k,N); 
            dS = sparse(k*k,N);
            dV = sparse(n*k,N);
            % cater for sparse columns in A.derivatives
            [~,jN] = find(A.derivatives); % find non-zero columns
            jN = unique(jN);
            N = length(jN);  % redefined to number of non-zero cols
            % use fast pagemtimes if available
            if exist('pagemtimes')==5,  Mx = @pagemtimes; 
            else Mx = @AutoDiff.pageMmult;
            end%if exist
            T = 'ctranspose';   O = 'none';
            dA = reshape(full(A.derivatives(:,jN)),m,n,N);
            dP = Mx(U,T,Mx(dA,V),O); % U'*dA*V  k.k.N
            iDiag = find(speye(k));
            dP2D = reshape(dP,k^2,N);
            dS(iDiag,jN) = real(dP2D(iDiag,:)); % diag(real(diag(dP)));
            dP = dP-reshape(full(dS(:,jN)),k,k,N);
            Foff=(1-eye(k))./(s'.^2-s.^2+eye(k)); % k.k
            symM=@(X) X+conj(permute(X,[2 1 3]));
            UdUoff=Foff.*symM(dP.*s'); % Foff.*(dP*S+S*dP'), k.k.N
            VdVoff=Foff.*symM( s.*dP); % Foff.*(S*dP+dP'*S), k.k.N
            dVi = Mx(V, VdVoff ); % V*( VdVoff ), n.k.N
            dP2D = reshape(dP,k^2,N);
            UdUoff = reshape(UdUoff,k^2,N);  % UdUoff+diag(diag(dP)./s)
            UdUoff(iDiag,:) = UdUoff(iDiag,:)+dP2D(iDiag,:)./s;
            dUi = Mx(U, reshape(UdUoff,k,k,N) ); % U*( UdUoff ), m.k.N
            if m>k, dUi=dUi+Mx(Mx(eye(m)-U*U',O,dA,O),V./s'); end
            if n>k, dVi=dVi+Mx(Mx(eye(n)-V*V',O,dA,T),U./s'); end
            c = nan(1,k,N);
            for j=1:k
                c(1,j,:) = (-1i./abs(V(iVM(j),j))).*imag(dVi(iVM(j),j,:));
            end%for j
            dV(:,jN) = reshape( dVi+V.*c ,n*k,N);
            dU(:,jN) = reshape( dUi+U.*c ,m*k,N);
            % form results as AutoDiff entities
            Uad = AutoDiff(U,dU);
            Sad = AutoDiff(S,dS);
            Vad = AutoDiff(V,dV);
        end% svd()

        function n = numel(x)
            n = numel(x.values);
        end

        function x = plus(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    x = repmat_as(x, y);
                    y = repmat_as(y, x);
                    x.values = x.values + y.values;
                    x.derivatives = x.derivatives + y.derivatives;
                else
                    x = repmat_as(x, y);
                    y = repmat_as(y, x);
                    y.values = x + y.values;
                    x = y;
                end
            else
                x = repmat_as(x, y);
                y = repmat_as(y, x);
                x.values = x.values + y;
            end
        end

        function x = power(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    temp = x.values.^(y.values);
                    x.derivatives = AutoDiff.spdiag(y.values.*x.values.^(y.values - 1)) * x.derivatives ...
                        +AutoDiff.spdiag(temp.*log(x.values)) * y.derivatives;
                    x.values = temp;
                else
                    y.values = x.^y.values;
                    y.derivatives = AutoDiff.spdiag(y.values.*log(x))* y.derivatives;
                    x = y;
                end
            else
                x.derivatives = AutoDiff.spdiag(y.*x.values.^(y - 1)) * x.derivatives;
                x.values = x.values.^y;
            end
        end

        function x = rdivide(x, y)
            if isa(y, 'AutoDiff')
                if isa(x, 'AutoDiff')
                    x = repmat_as(x, y);
                    y = repmat_as(y, x);
                    x.derivatives = AutoDiff.spdiag(1./y.values) * x.derivatives - AutoDiff.spdiag(x.values./y.values.^2) * y.derivatives;
                    x.values = x.values ./ y.values;

                else
                    x = repmat_as(x, y);
                    y = repmat_as(y, x);
                    y.derivatives = AutoDiff.spdiag(-x./y.values.^2) * y.derivatives;
                    y.values = x ./ y.values;
                    x = y;
                end
            else
                x = repmat_as(x, y);
                y = repmat_as(y, x);
                x.derivatives = AutoDiff.spdiag(1./y) * x.derivatives;
                x.values = x.values ./ y;
            end
        end

        function x = reshape(x, varargin)
            x.values = reshape(x.values, varargin{:});
        end

        function varargout = size(x, varargin)
            if nargin == 1

                if nargout <= 1
                    s = size(x.values);
                    varargout = {s};
                elseif nargout == 2
                    [sx, sy] = size(x.values);
                    varargout = {sx, sy};
                else
                    error('not yet coded')
                end
            else
                sx = size(x.values, varargin{:});
                varargout = {sx};
            end
        end

        
        function x = cumsum(x, varargin)
            val = cumsum(x.values, varargin{:});
 
            if isvector(x.values)
                [k, l] = meshgrid(1:numel(x), 1:numel(x));
                J = k<=l;
                x.derivatives = J*x.derivatives;
            else
                if (nargin > 1) && isscalar(varargin{1})
                    dim = varargin{1};
                else
                    dim = 1;
                end                
                [k, l] = meshgrid(1:size(x, dim), 1:size(x,dim));
                Jv = k<=l;
                sz = size(x.values);
                prodsz1 = prod(sz(1:dim-1));
                prodsz2 = prod(sz(dim+1:end));
                J = kron(speye(prodsz2), kron(Jv,speye(prodsz1)));
                x.derivatives = J*x.derivatives;
            end
            x.values = val;
        end
        
        function varargout = sort(x, varargin)
            [val, idx] = sort(x.values, varargin{:});


            if isvector(x.values)
                x.derivatives = x.derivatives(idx(:), :);
            elseif ismatrix(x.values)
                if (nargin > 1) && isscalar(varargin{1})
                    dim = varargin{1};
                else
                    dim = 1;
                end
                if dim == 1
                    idx2 = idx + (0:size(x.values, 2) - 1) * size(x.values, 1);
                else
                    idx2 = (idx - 1) * size(x.values, 1) + (1:size(x.values, 1))';
                end

                x.derivatives = x.derivatives(idx2(:), :);
            else
                error('not coded yet')
            end
            x.values = val;

            varargout{1} = x;
            if nargout > 1
                varargout{2} = idx;
            end
        end

        function y = subsasgn(y, S, x)
            if isempty(S.subs{1})
                return;
            end
            if isa(x, 'AutoDiff')

                tmp = reshape(1:numel(y), size(y));
                tmp(S.subs{:}) = zeros(size(x));
                [listwherey, ~, listkeepy] = find(tmp(:));
                tmp = zeros(size(y));
                tmp(S.subs{:}) = reshape(1:numel(x), size(x));
                listwherex = find(tmp);
                y.values(S.subs{:}) = x.values;


                if issparse(y.values)
                    warning('AutoDiff:Inefficient', 'this emplementation is quite inefficent')
                end

                if ~isa(y, 'AutoDiff')
                    y = AutoDiff(y.values, sparse(numel(y.values), size(x.derivatives, 2)));
                end

                %y.derivatives(tmp,:)=x.derivatives; % slow for some
                %reasons for large sparse matrices  

                n = numel(y.values);
                m = numel(x);
                y.derivatives = sparse(listwherey, listkeepy, ones(1, numel(listwherey)), n, size(y.derivatives, 1)) * y.derivatives + ...
                    +sparse(listwherex, 1:m, ones(1, m), n, m) * x.derivatives;
            else

                tmp = reshape(1:numel(y), size(y));
                tmp(S.subs{:}) = zeros(size(x));
                [listwherey, ~, listkeepy] = find(tmp(:));
                y.values(S.subs{:}) = x;
                if issparse(y.values)
                    warning('AutoDiff:Inefficient', 'this emplementation is quite inefficent')
                end
                n = numel(y.values);
                y.derivatives = sparse(listwherey, listkeepy, ones(1, numel(listwherey)), n, size(y.derivatives, 1)) * y.derivatives;
            end
        end

        function x = subsref(x, s)

            switch s(1).type
                case '()'
                    t = reshape(1:numel(x.values), size(x.values));
                    % TO DO: refactor as it might be very inefficient if x
                    % is sparse
                    if issparse(x.values)
                        warning('AutoDiff:Inefficient', 'this implementation is quite inefficent')
                    end

                    tsub = t(s.subs{:});
                    x.values = reshape(x.values(tsub), size(tsub));
                    if issparse(x.derivatives)
                        tmp = sparse(1:numel(tsub), tsub(:)', ones(1, numel(tsub)), numel(tsub), size(x.derivatives, 1));
                        x.derivatives = tmp * x.derivatives;
                    else
                        x.derivatives = x.derivatives(tsub(:), :);
                    end
                    if not(issparse(x.derivatives)) && (numel(x.derivatives ~= 0) < 0.1 * numel(x.derivatives))
                        x.derivatives = sparse(x.derivatives);
                    end
                case '.'
                    if length(s) > 1
                        x = x.(s(1).subs)(s(2).subs{:});
                    else
                        x = x.(s.subs);
                    end

                otherwise
                    error('Specify value for x as obj(x)')
            end
        end


        function x = sum(x, dim)
            if nargin == 1
                s = size(x.values);
                assert(length(s) == 2)
                if s(1) == 1
                    dim = 2;
                else
                    dim = 1;
                end
            end            
            
            if dim<=ndims(x.values) 
                sx = size(x.values);
                nin = numel(x.values);
                x.values = sum(x.values, dim);
                nout = numel(x.values);
                r = ones(sx(dim), 1) * (1:nout);
                c = permute(reshape(1:nin, sx), [dim, 1:dim - 1, dim + 1:numel(sx)]);
                x.derivatives = sparse(r(:), c(:), ones(1, nin), nout, nin) * x.derivatives;
            end
        end

        function x = mean(x, dim)
            s = size(x.values);
            if nargin == 1
                assert(length(s) == 2)
                if s(1) == 1
                    dim = 2;
                else
                    dim = 1;
                end
            end
            if dim<=ndims(x.values) 
                x = sum(x, dim) / s(dim);
            end
        end
    
        function z = repmat_to_size(x, target_size)
            if (ndims(x) == numel(target_size)) && all(size(x) ==target_size)
                z = x;
            elseif isa(x, 'AutoDiff')
                r = reshape((1:numel(x.values)), size(x.values)) .* ones(target_size);
                z.values = x.values(r);
                z.derivatives = sparse(1:numel(r), r(:), ones(1, numel(r))) * x.derivatives;
                z = AutoDiff(z);
            else
                r = reshape((1:numel(x)), size(x)) .* ones(target_size);
                z = x(r);
            end
        end
            
        function z = repmat_as(x, y)
            target_size = size(y); 
            if (ndims(x) == numel(target_size)) && all(size(x) ==target_size)
                z = x;
            elseif isa(x, 'AutoDiff')
                r = reshape((1:numel(x.values)), size(x.values)) .* ones(target_size);
                z.values = x.values(r);
                z.derivatives = sparse(1:numel(r), r(:), ones(1, numel(r))) * x.derivatives;
                z = AutoDiff(z);
            else
                r = reshape((1:numel(x)), size(x)) .* ones(target_size);
                z = x(r);
            end
        end

        function n = ndims(x)
            n = ndims(x.values);
        end
        function z = times(x, y)
            if isa(x, 'AutoDiff')
                if isa(y, 'AutoDiff')
                    z.values = x.values .* y.values;
                    if numel(x.values) == 1
                        z.derivatives = sparse(y.values(:)) * x.derivatives + x.values * y.derivatives;
                    elseif numel(y.values) == 1
                        z.derivatives = sparse(x.values(:)) * y.derivatives + y.values * x.derivatives;
                    elseif (ndims(x) == ndims(y)) && all(size(x) == size(y))
                        z.derivatives = AutoDiff.spdiag(y.values) * x.derivatives + AutoDiff.spdiag(x.values) * y.derivatives;
                    else %using broadcasting
                        x = repmat_as(x, y);
                        y = repmat_as(y, x);
                        z.derivatives = AutoDiff.spdiag(y.values) * x.derivatives + AutoDiff.spdiag(x.values) * y.derivatives;
                    end

                else
                    z.values = x.values .* y;
                    if numel(x.values) == 1
                        z.derivatives = sparse(y(:)) * x.derivatives;
                    else
                        if (ndims(x) == ndims(y)) && all(size(x) == size(y))
                            z.derivatives = AutoDiff.spdiag(y) * x.derivatives;
                        else %using broadcasting
                            x = repmat_as(x, y);
                            y = repmat_as(y, x);
                            z.derivatives = AutoDiff.spdiag(y) * x.derivatives;
                        end
                    end
                end
                z = AutoDiff(z);
            else
                z = times(y, x);
            end

        end


        function [V, D] = eig(C)
            % Compute the eigen vector  eigen values and there derivative with respect
            % to each element of the input matrix. The function might be undifferentiable
            % if the mutiplicity of an eigen value is more than one.
            % It may no work if C is not symmetric (need to check if the formulas are still valid)
            if any(any(C.values' - C.values) > eps)
                error('not yet verified for non symetric matrices')
            end

            n = size(C, 1);
            [V, D] = eig(C.values);
            lambda = diag(D);
            % C.values*V==V*D
            % k=1
            % C.values*V(:,k)=lambda(k)*V(:,k)
            %

            l = 0;

            dV_dC = zeros(n, n, n^2);
            dD_dC = zeros(n, n, n^2);

            dlambda = zeros(size(C, 1), n^2);
            for j = 1:n
                for i = 1:n
                    l = l + 1;

                    Ap = sparse(i, j, 1, size(C, 1), size(C, 1));


                    for k = 1:size(C, 1)
                        %dlambda(k,l)=V(i,k)*V(j,k)
                        dlambda(k, l) = V(:, k)' * Ap * V(:, k);


                        % B=[C-lambda(k)*eye(n,n);V(:,k)'];
                        % dV_dC(:,k,l)=(B'*B)^-1*B'*[dlambda(k)*V(:,k)-Ap*V(:,k);0];
                        dV_dC(:, k, l) = [C.values - lambda(k) * eye(n, n); V(:, k)'] \ [dlambda(k, l) * V(:, k) - Ap * V(:, k); 0];
                        % [C-lambda(k)*eye(3,3)]*dV_dC(:,k,l)+-dlambda(k)*V


                        %n=size(C.values,1);
                        %  k=1;
                        %
                        % (C.values-lambda(k)*eye(n))*V(:,k)

                        %   Ap=sparse(i,j,1,size(C,1),size(C,1));
                        %   (Ap-dlambda(k,l)*eye(n))*V(:,k)+(C.values-lambda(k)*eye(n))*dV_dC(:,k,l)
                        %  dV_dC(:,k,l)'*V(:,k)
                        %  V(:,k)'*(Ap-dlambda(k,l)*eye(n))*V(:,k)+V(:,k)'*(C.values-lambda(k)*eye(n))*   dV_dC(:,k,l)

                        %   V(:,k)'*(C.values-lambda(k)*eye(n))
                    end
                    dD_dC(:, :, l) = diag(dlambda(:, l));
                end
            end


            if nargout == 1
                V = AutoDiff(lambda, dlambda);
            else

                D = AutoDiff(D, reshape(dD_dC, numel(D), [])*C.derivatives);
                V = AutoDiff(V, reshape(dV_dC, numel(D), [])*C.derivatives);
            end
        end


        function x = transpose(x)
            M = AutoDiff.transposeDiff(size(x));
            x.derivatives = M * (x.derivatives);
            x.values = x.values';
        end

        function x = permute(x, l)
            t = reshape(1:numel(x.values), size(x.values));
            t = permute(t, l);
            x.values = permute(x.values, l);
            x.derivatives = sparse(1:numel(t), t(:), ones(1, numel(t))) * x.derivatives;

        end

        function x = uminus(x)
            x.values = -x.values;
            x.derivatives = -x.derivatives;
        end

        function x = uplus(x)
        end

        function y = horzcat(varargin)
            y = cat(2, varargin{:});
        end


        function y = det(x)
            y.values = det(x.values);
            y.derivatives = reshape(det(x.values).*inv(x.values)',1,[]) * x.derivatives;
            y = AutoDiff(y.values, y.derivatives);
        end

        function y = vertcat(varargin)
            y = cat(1, varargin{:});
        end

        function x = ones(varargin)
            k = find(cellfun(@isnumeric, varargin), 1, 'last');
            assert(length(varargin) == k+2);
            assert(strcmp(varargin{k + 1}, 'like'));
            x.values = ones(varargin{1:k});
            x.derivatives = sparse(numel(x.values), size(varargin{k + 2}.derivatives, 2));%change zeros to sparse, AJR 3/6/26
            x = AutoDiff(x);
        end

        function x = zeros(varargin)
            k = find(cellfun(@isnumeric, varargin), 1, 'last');
            assert(length(varargin) == k+2);
            assert(strcmp(varargin{k + 1}, 'like'));
            x.values = zeros(varargin{1:k});
            x.derivatives = sparse(numel(x.values), size(varargin{k + 2}.derivatives, 2));%changed zeros to sparse, AJR 3/6/26
            x = AutoDiff(x);
        end

        function x = nan(varargin)
        % nan(...) for AutoDiff-likeness gives nan-matrix according
        % to varargin, and sparse-zero derivatives. AJR, 9 Jun 2026 
            k = find(cellfun(@isnumeric, varargin), 1, 'last');
            assert(length(varargin) == k+2);
            assert(strcmp(varargin{k + 1}, 'like'));
            x.values = nan(varargin{1:k});
            x.derivatives = sparse(numel(x.values) ...
                ,size(varargin{k + 2}.derivatives, 2));
            x = AutoDiff(x);
        end

        function r = rank(x, varargin)
            r = rank(x.values, varargin{:});
        end

    end

    methods (Static)


        function C = pageMmult(A,At,B,Bt)
        % Computes matrix product A*B for ensemble of matrices. 
        % Arguments are either two (A,B), or four (A,At,B,Bt), where
        % A & B are 3-D arrays.  For each l computes
        % A(:,:,l)*B(:,:,l) where A&B may be transposed according to
        % At & Bt: 'none' (default), 'ctranspose' or 'transpose' 
        % AJR, 25 Jun 2026
        if nargin==2, B=At; At='none'; Bt='none'; 
        else assert(nargin==4 ...
            ,'AutoDiff.pageMmult: must have 2 or 4 arguments')
        end%if
        [a1,a2,a3]=size(A); [b1,b2,b3]=size(B);
        k = max(a3,b3);
        if a3>1, a=@(l) l; else a=@(l) 1; end
        if b3>1, b=@(l) l; else b=@(l) 1; end
        switch At
        case 'none',       opA=@(X) X;      c1=a1;
        case 'ctranspose', opA=@ctranspose; c1=a2;
        case 'transpose',  opA=@transpose;  c1=a2;
        otherwise error('AutoDiff.pageMmult: unknown A-transpose')
        end%switch At
        switch Bt
        case 'none',       opB=@(X) X;      c2=b2;
        case 'ctranspose', opB=@ctranspose; c2=b1;
        case 'transpose',  opB=@transpose;  c2=b1;
        otherwise error('AutoDiff.pageMmult: unknown B-transpose')
        end%switch Bt
        C=nan(c1,c2,k);
        for l=1:k, C(:,:,l)=opA(A(:,:,a(l)))*opB(B(:,:,b(l))); end
        end%function pageMmult


        function M = spDiagFromVec(v)
            M = sparse((1:numel(v)), (1:numel(v)), v(:));
        end


        function d = spdiag(a)
            if isscalar(a)
                d = a;
            elseif issparse(a)
                [t, ~, v] = find(a(:));
                n = numel(a);
                d = sparse(t, t, v, n, n);
            else
                d = sparse(1:numel(a), 1:numel(a), a(:));
            end
        end


        function D = transposeDiff(sizeM)
            listJ = repmat((0:sizeM(2)-1)'*sizeM(1), 1, sizeM(1)) + repmat((1:sizeM(1)), sizeM(2), 1);
            D = sparse(1:prod(sizeM), listJ, ones(1, prod(sizeM)));
        end

        function M = subscriptDiff(idx, idy, sizeB)

            if numel(sizeB) ~= 2
                error('sizeB should be of size 2')
            end
            nout = numel(idx) * numel(idy);
            listJ = repmat(idx(:), [1, numel(idy)]) + repmat(sizeB(1)*(idy(:)' - 1), [numel(idx), 1]);
            M = sparse(1:nout, listJ(:), ones(1, numel(idx) * numel(idy)), nout, prod(sizeB));
        end
    end
end
