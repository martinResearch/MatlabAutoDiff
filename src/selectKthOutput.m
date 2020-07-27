function y = selectKthOutput(n, k, f, x, varargin)

% selectKthOutput returns the kth output of the function f evaluated at x
% with n outputs. Can be of handy to write anonymous functions on one single line
%
%
%  y=AutoDiffSelectOutput(n,f,x)
%
% Inputs:
%     n       : number of ouput the function is expected to return
%     k       : integer index of the output we want to obtain
%     f       : handle to the function
%     x       : entry of the function
% Outputs:
%     y       : nth output of the function f evaluated at x
%
% Description:
%     returns the nth output of the function f evaluated at x
%
%Example:
%  f=@(x) sort(diag(selectKthOutput(2,2,@eig,x)));
% will return the sorted eigen values of the matrix x
%
% Documentation created  by Martin de La Gorce


if n == 1
    y = f(x);
else

    eval(['[', repmat('~,', 1, k - 1), 'y', repmat(',~', 1, n - k), ']=f(x,varargin{:});']);
    % eval(['[',repmat('~,',1,n-1),'y]=f(x,varargin{:});']); if matlab>=2010
end