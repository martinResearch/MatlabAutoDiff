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


classdef AutoDiffVerif

    %   Class that allows to compare result obtained with AutoDiffFD and AutoDiff
    %   used in AutoDiffJacobian when mode='VerifiedAutoDiff'' and there is
    %   a discrependy between the finite differences method and the
    %   automatic differentiation method. This class helps in indetify
    %   which operation does not have the right jacobian in the
    %   computational graph.
    %   Documentation created  by Martin de La Gorce


    properties
        values;
        AD;
        FD;
    end

    methods

        function x = AutoDiffVerif(values, AD, FD)
            if isa(values, 'struct')
                x = values;
                x = AutoDiffVerif(x.values, x.AD, x.FD);
            elseif nargin == 1
                x.values = values;
                x.AD = AutoDiff(values);
                x.FD = AutoDiffFD(values);
            elseif nargin == 3
                x.values = values;
                x.AD = AD;
                x.FD = FD;
            else
                error('not expected');
            end


        end

        function Jac = getderivsAD(x)

            Jac = getderivs(x.AD);
        end

        function Jac = getderivsFD(x)
            Jac = getderivs(x.FD);
        end

        function val = getvalue(x)
            val = x.values;
        end

        function val = getvalueAD(x)
            val = getvalue(x.AD);
        end

        function val = getvalueFD(x)
            val = getvalue(x.FD);
        end

        function x = setderivs(x, derivatives)
            x.derivatives = derivatives;
        end
        function double(~)


            error(['Conversion to double from AutoDiff is not possible.', ...
                ' This my be due to preallocation of the left side of the assignement.', ...
                ' Considere modifying the code to avoid preallocation of converting the', ...
                ' preallocated array to AutoDiff by multiplying it with an AutoDiff variable.', ...
                ' As an example b=AutoDiff(1);a=[0,0];a(1)=b;  could be replaced by  b=AutoDiff(1);a=[0,0]*b(1);a(1)=b']);
        end


        function x = fevalSingle(func, x)
            x.values = func(x.values);
            x.AD = func(x.AD);
            x.FD = func(x.FD);


            checkConsistency(x)

        end


        function y = fevalMultivariate(func, varargin)
            list = varargin;
            y = fevalMultivariateVarargin(@(x) func(x{:}), list);
        end

        function x = subsindex(x)
            x = x.values;
        end

        function x = abs(x)
            x = fevalSingle(@abs, x);
        end
        function x = sign(x)
            x = sign(x.values);
        end

        function x = acos(x)
            x = fevalSingle(@acos, x);
        end

        function x = asin(x)
            x = fevalSingle(@asin, x);
        end

        function x = atan(x)
            x = fevalSingle(@atan, x);
        end

        function x = ceil(x)
            x = fevalSingle(@ceil, x);
        end


        function x = sin(x)
            x = fevalSingle(@sin, x);
        end

        function x = cos(x)
            x = fevalSingle(@cos, x);
        end

        function x = conj(x)
            x = fevalSingle(@conj, x);
        end

        function x = isreal(x)
            x = fevalSingle(@isreal, x);
        end

        function x = ctranspose(x)
            x = fevalSingle(@ctranspose, x);
        end

        function x = not(x)
            x = fevalSingle(@not, x);
        end

        function x = exp(x)
            x = fevalSingle(@ctranspose, x);
        end

        function x = floor(x)
            x = fevalSingle(@floor, x);
        end

        function x = log(x)
            x = fevalSingle(@log, x);
        end


        function x = sqrt(x)
            x = fevalSingle(@sqrt, x);
        end


        function x = tan(x)
            x = fevalSingle(@tan, x);
        end

        function x = tanh(x)
            x = fevalSingle(@tanh, x);
        end


        function y = cat(dim, varargin)
            list = varargin;
            y = fevalMultivariateVarargin(@(x) cat(dim, x{:}), list);
        end


        function x = diag(x)
            x = fevalSingle(@diag, x);
        end

        function varargout = sort(x)
            varargout = fevalSingleVarargout(@sort, x, nargout);
        end

        function x = diff(x, n, dim)
            f = @(x) diff(x, n, dim);
            x = fevalSingle(f, x);
        end

        function idx = end (x, k, n)
            if k == 1 && n == 1
                idx = length(x.values);
                return
            end
            idx = size(x.values, k);
        end

        function z = eq(x, y)
            if isa(y, 'AutoDiffVerif')
                if isa(x, 'AutoDiffVerif')
                    z = x.values == y.values;
                else
                    z = x == y.values;
                end
            else
                z = x.values == y;
            end
        end


        function z = ne(x, y)

            if isa(y, 'AutoDiffVerif')
                if isa(x, 'AutoDiffVerif')
                    z = x.values ~= y.values;
                else
                    z = x ~= y.values;
                end
            else
                z = x.values ~= y;
            end
        end


        function z = ge(x, y)
            if isa(y, 'AutoDiffVerif')
                if isa(x, 'AutoDiffVerif')
                    z = x.values >= y.values;
                else
                    z = x >= y.values;
                end
            else
                z = x.values >= y;
            end
        end

        function z = gt(x, y)
            if isa(y, 'AutoDiffVerif')
                if isa(x, 'AutoDiffVerif')
                    z = x.values > y.values;
                else
                    z = x > y.values;
                end
            else
                z = x.values > y;
            end
        end


        function z = lt(x, y)
            if isa(y, 'AutoDiffVerif')
                if isa(x, 'AutoDiffVerif')
                    z = x.values < y.values;
                else
                    z = x < y.values;
                end
            else
                z = x.values < y;
            end
        end


        function z = le(x, y)
            if isa(y, 'AutoDiffVerif')
                if isa(x, 'AutoDiffVerif')
                    z = x.values <= y.values;
                else
                    z = x <= y.values;
                end
            else
                z = x.values <= y;
            end
        end

        function y = isnan(x)
            y = isnan(x.values);
        end


        function mylength = length(x)
            mylength = length(x.values);
        end


        function y = horzcat(varargin)
            y = cat(2, varargin{:});
        end
        function y = max(x)
            y = fevalSingle(@max, x);
        end

        function y = min(x)
            y = fevalSingle(@min, x);
        end

        function x = minus(x, y)
            x = fevalMultivariate(@minus, x, y);
        end

        function x = mpower(x, y)
            x = fevalMultivariate(@mpower, x, y);
        end

        function inv(~)
            error('not coded yet')
        end


        function z = mldivide(x, y)
            z = fevalMultivariate(@mldivide, x, y);
        end

        function z = mrdivide(x, y)

            z = fevalMultivariate(@mrdivide, x, y);
        end


        function z = mtimes(x, y)
            z = fevalMultivariate(@mtimes, x, y);
        end

        function x = norm(x, p)
            x = fevalMultivariate(@norm, x, p);
        end

        function x = plus(x, y)
            x = fevalMultivariate(@plus, x, y);
        end

        function x = power(x, y)
            x = fevalMultivariate(@power, x, y);
        end

        function x = rdivide(x, y)
            x = fevalMultivariate(@rdivide, x, y);
        end

        function x = reshape(x, varargin)
            x = fevalSingle(@(x) reshape(x, varargin{:}), x);

        end

        function z = convhull(x, y)
            z = fevalMultivariate(@convhull, x, y);
            z = z.values;
        end


        function varargout = size(x, varargin)
            if nargin == 1
                [sx, sy] = size(x.values);
                if nargout <= 1
                    varargout = {[sx, sy]};
                else
                    varargout = {sx, sy};
                end
            else
                sx = size(x.values, varargin{:});
                varargout = {sx};
            end
        end


        function n = numel(x)
            n = numel(x.values);
        end


        function y = subsasgn(y, S, x)
            y = fevalMultivariate(@(x, y) subsasgn(y, S, x), x, y);
        end

        function x = subsref(x, s)
            x = fevalSingle(@(x) subsref(x, s), x);
        end


        function x = sum(x, dim)
            if nargin == 1
                x = fevalSingle(@sum, x);
            else
                x = fevalSingle(@(x)sum(x, dim), x);
            end
        end


        function z = times(x, y)
            z = fevalMultivariate(@times, x, y);
        end

        function x = transpose(x)
            x = fevalSingle(@transpose, x);
        end

        function x = permute(x, l)
            x = fevalSingle(@(x) permute(x, l), x);

        end

        function x = uminus(x)
            x = fevalSingle(@uminus, x);
        end

        function x = uplus(x)
            x = fevalSingle(@uplus, x);
        end

        function y = vertcat(varargin)
            y = cat(1, varargin{:});
        end

        function x = rank(x, varargin)
            x = fevalSingle(@(x) rank(x, varargin{:}), x);
        end
        function x = det(x)
            x = fevalSingle(@det, x);
        end

        function varargout = eig(C)
            error('eig is not differencial numerically unless you sort the output, use eigsorted instead');
        end
        function varargout = eigsorted(C)
            out = fevalSingleVarargout(@eigsorted, C, max(nargout, 1));
            varargout = out;
        end
        function x = zeros(x)
            error('is it needed ?')
            x.values = x.values * 0;
            x.derivatives = x.derivatives * 0;
        end


    end

end


function out = fevalSingleVarargout(func, x, N)


tmp = AutoDiffMultipleOutput(N, func, x.values);
tmpAD = AutoDiffMultipleOutput(N, func, x.AD);
tmpFD = AutoDiffMultipleOutput(N, func, x.FD);

out = cell(1, N);
for k = 1:N
    out{k}.values = tmp{k};
    out{k}.AD = tmpAD{k};
    out{k}.FD = tmpFD{k};
    out{k} = AutoDiffVerif(out{k});
end


checkConsistencyCell(out)

end

function y = fevalMultivariateVarargin(func, list)


listValues = cell(size(list));
listAD = cell(size(list));
listFD = cell(size(list));
for k = 1:numel(list)
    x = list{k};
    if isa(x, 'AutoDiffVerif')
        listValues{k} = x.values;
        listAD{k} = x.AD;
        listFD{k} = x.FD;
    else
        listValues{k} = x;
        listAD{k} = x;
        listFD{k} = x;
    end
end


y.values = func(listValues);
y.AD = func(listAD);
y.FD = func(listFD);
y = AutoDiffVerif(y);
checkConsistency(y)
end

function checkConsistency(x)
if ~isa(x.AD, 'AutoDiff')
    if isa(x.FD, 'AutoDiffFD')

        JFD = getderivsFD(x);
        if any(JFD(:))
            error('JFD should be empty or equal to zero')
        end
    end
else


    JAD = getderivsAD(x);
    JFD = getderivsFD(x);

    if any((JAD(:) ~= 0) & (JFD(:) == 0))
        warning('zeros in the derivatives obtained with FD should be present in the one obtained with AD');
    end

    if norm(JAD-JFD) > 1e-3
        warning('difference too large');
    end
end
end

function checkConsistencyCell(x)
for k = 1:numel(x)
    checkConsistency(x{k});
end
end
