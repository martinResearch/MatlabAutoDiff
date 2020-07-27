% License FreeBSD:
%
% Copyright (c) 2016  Martin de La Gorce
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this
%    list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice,
%    this list of conditions and the following disclaimer in the documentation
%    and/or other materials provided with the distribution.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
% ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%
% The views and conclusions contained in the software and documentation are those
% of the authors and should not be interpreted as representing official policies,
% either expressed or implied, of the FreeBSD Project.


classdef AutoDiff

    %
    %    This class implement a forward automatic differentation method based
    %    on operator overloading. This class allows precise and efficient
    %    computation of function Jacobians by calling AutoDiffJacobianAutoDiff
    %
    %   In contrast with most AD matlab tools
    %    - Derivatives are represented as sparse matrices
    %    - N dimensional array are supported
    %
    %   The speed could be improved by representing jacobian matrices by
    %   their transposed matrix , due to the way matlab store sparse matrices
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


        function x = abs(x)
            x.derivatives = AutoDiff.spdiag(sign(x.values)) * x.derivatives;
            x.values = abs(x.values);
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


        function x = atan(x)
            x.derivatives = AutoDiff.spdiag(1./(1 + x.values.^2)) * x.derivatives;
            x.values = atan(x.values);
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
                    y.values = cat(dim, y.values, x);
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
            x.derivatives = sparse(1:numel(r), r, ones(size(r))) * x.derivatives;
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
            if n ~= 1
                error('not yet coded')
            end

            t = reshape(1:numel(x.values), size(x.values));
            if issparse(x.values)
                warning('AutoDiff:Inefficient', 'this implementation is quite inefficent')
            end

            if ndims(x.values) ~= 2
                error('not yet coded')
            end

            if dim == 1

                tsub1 = t(2:end, :);
                tsub2 = t(1:end-1, :);
                D = sparse(1:numel(tsub1), tsub1(:)', ones(1, numel(tsub1)), numel(tsub1), size(x.derivatives, 1)) - ...
                    sparse(1:numel(tsub2), tsub2(:)', ones(1, numel(tsub2)), numel(tsub1), size(x.derivatives, 1));
                x.values = reshape(D*x.values(:), size(tsub1));
                x.derivatives = D * x.derivatives;

            elseif dim == 2

                tsub1 = t(:, 2:end, :);
                tsub2 = t(:, 1:end-1);
                D = sparse(1:numel(tsub1), tsub1(:)', ones(1, numel(tsub1)), numel(tsub1), size(x.derivatives, 1)) - ...
                    sparse(1:numel(tsub2), tsub2(:)', ones(1, numel(tsub2)), numel(tsub2), size(x.derivatives, 1));
                x.values = reshape(D*x.values(:), size(tsub1));
                x.derivatives = D * x.derivatives;
            else
                error('not yet coded')
            end

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


        function [m, id] = max(C, varagin)
            if nargin == 1
                if isvector(C.values)
                    [~, id] = max(C.values);
                    m = AutoDiff(C.values(id), C.derivatives(id, :));
                else
                    [v, id] = max(C.values);
                    id2 = id(:)' + (0:numel(id) - 1) * size(C.values, 1);
                    m.values = v;
                    tmp = sparse(1:numel(id2), id2(:)', ones(1, numel(id2)), numel(id2), size(C.derivatives, 1));
                    m.derivatives = tmp * C.derivatives;
                end
            elseif nargin == 2
                B = varagin(1);
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

        function [m, id] = min(C, varagin)
            if nargin == 1
                if isvector(C.values)
                    [~, id] = min(C.values);
                    m = AutoDiff(C.values(id), C.derivatives(id, :));
                else
                    [v, id] = min(C.values);
                    id2 = id(:)' + (0:numel(id) - 1) * size(C.values, 1);
                    m.values = v;
                    tmp = sparse(1:numel(id2), id2(:)', ones(1, numel(id2)), numel(id2), size(C.derivatives, 1));
                    m.derivatives = tmp * C.derivatives;
                end
            elseif nargin == 2
                B = varagin(1);
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

        function [U, S, V] = svd(x)
            error('not coded yet, could look at the eig implementation')
        end

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
                    y.derivatives = AutoDiff. valXder(y.values.*log(x), y.derivatives);
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
            sx = size(x.values);
            nin = numel(x.values);
            x.values = sum(x.values, dim);
            nout = numel(x.values);
            r = ones(sx(dim), 1) * (1:nout);
            c = permute(reshape(1:nin, sx), [dim, 1:dim - 1, dim + 1:numel(sx)]);
            x.derivatives = sparse(r(:), c(:), ones(1, nin), nout, nin) * x.derivatives;
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
            x = sum(x, dim) / s(dim);
        end

        function z = repmat_as(x, y)
            if (ndims(x) == ndims(y)) && all(size(x) == size(y))
                z = x;
            elseif isa(x, 'AutoDiff')
                r = reshape((1:numel(x.values)), size(x.values)) .* ones(size(y));
                z.values = x.values(r);
                z.derivatives = sparse(1:numel(r), r(:), ones(1, numel(r))) * x.derivatives;
                z = AutoDiff(z);
            else
                r = reshape((1:numel(x)), size(x)) .* ones(size(y));
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
            if all(size(x) == [2, 2])
                y.values = det(x.values);
                y.derivatives = [x.values(2, 2), -x.values(1, 2), -x.values(2, 1), x.values(1, 1)] * x.derivatives;
                y = AutoDiff(y.values, y.derivatives);
            else
                error('not yet coded')
            end
        end

        function y = vertcat(varargin)
            y = cat(1, varargin{:});
        end

        function x = ones(varargin)
            k = find(cellfun(@isnumeric, varargin), 1, 'last');
            assert(length(varargin) == k+2);
            assert(strcmp(varargin{k + 1}, 'like'));
            x.values = ones(varargin{1:k});
            x.derivatives = zeros(numel(x.values), size(varargin{k + 2}.derivatives, 2));
            x = AutoDiff(x);
        end

        function x = zeros(varargin)
            k = find(cellfun(@isnumeric, varargin), 1, 'last');
            assert(length(varargin) == k+2);
            assert(strcmp(varargin{k + 1}, 'like'));
            x.values = zeros(varargin{1:k});
            x.derivatives = zeros(numel(x.values), size(varargin{k + 2}.derivatives, 2));
            x = AutoDiff(x);
        end

        function r = rank(x, varargin)
            r = rank(x.values, varargin{:});
        end

    end

    methods (Static)

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
