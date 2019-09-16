function CheckAutoDiffJacobian(func,x,tol)     
[J2,f2]=AutoDiffJacobianFiniteDiff(func,x,1:numel(x),[]);
[J1,f1]=AutoDiffJacobianAutoDiff(func,x);

err_f=norm(f1-f2);
assert(err_f<tol);
err_J=norm(J1-J2);
assert(err_J<tol);