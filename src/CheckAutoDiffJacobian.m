function CheckAutoDiffJacobian(func,x,tol)  
% check the function is deterministic

assert(maxabsdiff(func(x),func(x))==0)
% check jacobian
[J2,f2]=AutoDiffJacobianFiniteDiff(func,x,1:numel(x),[]);
[J1,f1]=AutoDiffJacobianAutoDiff(func,x);

err_f=maxabsdiff(f1,f2);
assert(err_f<tol);
err_J=maxabsdiff(J1,J2);
assert(err_J<tol);

function m=maxabsdiff(a,b)
diff=a-b;
m=max(abs(diff(:)));
