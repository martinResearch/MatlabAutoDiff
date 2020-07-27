function [V, D] = eigsorted(C)
% Compute the eigen vector and eigen values and sort
% from the highest to the smallest eigen value
% this function should be use instead of eig that
% may provide eigen values in an arbitrary order, wich
% makes it not really differentiable
% Martin de La Gorce
% Ecole Centrale de Paris

n = size(C, 1);
[V, D] = eig(C);
[lambda, id] = sort(diag(D), 'descend');

D = diag(lambda);
V = V(:, id);
idrevert = V(1, :) < 0;
V(:, idrevert) = -V(:, idrevert);
