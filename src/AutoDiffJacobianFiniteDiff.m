function [J, f] = AutoDiffJacobianFiniteDiff(func, x, range, epsilons, centered)

% AutoDiffJacobianFiniteDiff returns the jacobian of function f evaluated at x using Finite
% Differences.
%
%
%  J = AutoDiffJacobianFiniteDiff(func,x)
%  J = AutoDiffJacobianFiniteDiff(func,x,range)
%
% Inputs:
%     func    : handle to the function
%     x       : location where the jacobian is evaluated
%     range   : (optional) vector of integer indices in [1,numel(x)]
% Outputs:
%     J       : jacobian matrix of function f evaluated at x
%
% Description:
%     returns the jacobian matrix of function f evaluated at x
%     if a range is specified , only the columns corresponding to the range
%     indices are computed. The computation is donc using Finite
%     Differences. This might be unprecise due to rounoff errors.
%     This is provided as a way to check the validity of the Automatic differenciation
%     method. Considere using AutoDiffJacobianAutoDiff instead to get faster
%     and more accurate derivatives.
%
% Documentation created  by Martin de La Gorce


if nargin < 3
    range = (1:numel(x));
end
if nargin < 5
    centered = true;
end
if nargin < 4 || isempty(epsilons)
    epsilons = ones(size(x)) .* 1e-6;
end
if isempty(range)
    range = (1:numel(x));
end

f = func(x);
J = zeros(numel(f), numel(range));
if centered
    for k = range(:)'
        x2 = x;
        x2(k) = x(k) + epsilons(k);
        f2 = func(x2);
        x2 = x;
        x2(k) = x(k) - epsilons(k);
        f3 = func(x2);
        J(:, k) = (f2(:) - f3(:)) / (2 * epsilons(k));
    end
else
    for k = range(:)'
        x2 = x;
        x2(k) = x(k) + epsilons(k);
        f2 = func(x2);

        J(:, k) = (f2(:) - f(:)) / (epsilons(k));
    end
end
