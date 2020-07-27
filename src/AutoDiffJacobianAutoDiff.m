function [J, f] = AutoDiffJacobianAutoDiff(func, x, range)

% AutoDiffJacobianAutoDiff returns the jacobian of function evaluated at x using Automatic
% Differentiation
%
%
%  J = AutoDiffJacobianAutoDiff(func,x)
%  J = AutoDiffJacobianAutoDiff(func,x,range)
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
%     indices are computed. The computation is donc using Automatic
%     Differentiation (Not to be confounded with Finite Differences method)
%     This is numerically more precise and, if the Jacobian is large and
%     sparse this might be much faster (see examples)
%
% Examples:
%
% % Simple example
%
% f=@(x) [sin(x(1)) cos(x(2)) tan(x(1)*x(2))]
% full(AutoDiffJacobianAutoDiff(f,[1,1]))
% AutoDiffJacobianFiniteDiff(f,[1,1])
%
% %Speedup illustration
%
% f=@(x) (log(x(1:end-1))-tan(x(2:end)))
% tic; JAD=AutoDiffJacobianAutoDiff(f,0.5*ones(1,5000));timeAD=toc;
% tic; JFD=sparse(AutoDiffJacobianFiniteDiff(f,0.5*ones(1,5000)));timeFD=toc;
% fprintf('speedup AD vs FD = %2.1f\n',timeFD/timeAD)
% fprintf('max abs difference = %e\n',full(max(abs(JAD(:)-JFD(:)))));
%
% % N-D array support
%
% f=@(x) sum(x.^2,3)
%
% Documentation created  by Martin de La Gorce
%


if nargin < 3
    xAD = AutoDiff(x);
    nr = numel(x);
else
    if max(range) > numel(x)
        error('out of range')
    end
    sx = size(x);
    nr = numel(range);
    xAD = AutoDiff(x, sparse(range, (1:nr), ones(1, nr), sx(1) * sx(2), nr));
end

try
    fAD = func(xAD);
    f = getvalue(fAD);
    J = getderivs(fAD);
catch exception
    warning('failed while calling the function with the AutoDiff instance, trying to call it with the plain data instead to chech that works')
    func(x);
    warning('It seems like to original function is ok with the plain data , The class AutoDiff needs debugging')
    % rethrow(exception)
    [~] = func(xAD); % better to call again the function instead of using exception as it makes it possible to use matlab's stop-if -error debugging functionality
end


if size(J, 2) ~= nr
    J = reshape(J, [], nr);
end
