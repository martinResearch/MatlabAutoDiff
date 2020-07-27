function [J, f] = AutoDiffJacobian(func, x, method, varargin)
% AutoDiffJacobian returns the jacobian of function f evaluated at x using either
% Automatic Differentiation or finite differences, or both for comparison
%
%  J = AutoDiffJacobian(func,x)
%  J = AutoDiffJacobian(func,x,method)
%  J = AutoDiffJacobian(func,x,method)GradEpsilons
%
% Inputs:
%     func    : handle to the function
%     x       : location where the jacobian is evaluated
%     method  : can be 'AutoDiff','FiniteDiff' or 'VerifiedAutoDiff'
% Outputs:
%     J       : jacobian matrix of function f evaluated at x
%
% Description:
%     returns the jacobian matrix of function f evaluated at x
%     uses
%        - automatic differentiation if method=='AutoDiff'
%        - finite differences if method=='AutoDiff'
%	 - both automatic differentiation and differences and compare the two results if method='VerifiedAutoDiff'
%
% Documentation created  by Martin de La Gorce


if nargin < 3
    method = 'AutoDiff';
end
if nargin < 4
    varargin = {};
end

assert(isa(func, 'function_handle'))

args = ParseArguments(varargin, 'GradEpsilons', []);

if strcmpi(method, 'AutoDiff')
    [J, f] = AutoDiffJacobianAutoDiff(func, x);
elseif strcmpi(method, 'FiniteDiff')
    [J, f] = AutoDiffJacobianFiniteDiff(func, x, 1:length(x), args.GradEpsilons);
elseif strcmpi(method, 'VerifiedAutoDiff')
    [J, f] = AutoDiffJacobianAutoDiff(func, x);
    [J2, ~] = AutoDiffJacobianFiniteDiff(func, x, 1:numel(x), args.GradEpsilons);

    err = norm(J-J2);
    fprintf('difference between finite differencing and automatic differencing = %e\n', err);
    if any((J2(:) == 0) & (J(:) ~= 0))
        warning('this is VERY suspicius (zeros in the finite diff jacobian that are not in the AD jacobian), we check that there is no bug in the optAD class by calling the function with the AutoDiffveriv class...');
        xADv = AutoDiffVerif(x);
        fADx = func(xADv);
    end
    if err > 1e-4
        figure; imagesc((J2 == 0) & (J ~= 0))
        warning('this is suspicius, we check that there is no bug in the optAD class by calling the function with the AutoDiffVerif class...');
        xADv = AutoDiffVerif(x);
        fADx = func(xADv);

    end
    %
    % figure;imagesc((J2==0)&(J3~=0))
    % norm(J4-J2)
else
    error('unknown differentiation method')
end
