function v2 = autodiff_identity(v, x)
% returns the first argument if x is a double or ouble array
% returns an Autodiff instance if x is an Autodiff instance, with v2
% derivatives matrix with the same number of columns as the derivatives
% matrix of x
v2 = v + 0 * x(1);
end
