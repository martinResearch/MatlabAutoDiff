function [f, J] = AutoDifffuncAndJac(func, x, method, varargin)
% Provides both the value and the jacobian of the function at x
% similar to AutoDiffJacobian but provides the function value before the jacobian
% in the list of outputs, which can be usefull when using the matlab optimization toolbox
if nargin < 3
    method = 'AutoDiff';
end
if nargin < 4
    varargin = {};
end
args = ParseArguments(varargin, 'GradEpsilons', []);

if nargout == 2
    [J, f] = AutoDiffJacobian(func, x, method, 'GradEpsilons', args.GradEpsilons);
else
    f = func(x);
end
