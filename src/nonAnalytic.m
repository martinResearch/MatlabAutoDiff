function [flag] = nonAnalytic(fn,z)
% classifies whether a coded function fn is analytic or not
% at the complex point / vector / matrix z. Tests whether fn
% is analytic via auto-differentiation of the fn as provided
% by the package AutoDiff.  Reports flag=0 if fn satisfies
% the Cauchy--Riemann eqns;  =1 if fn fails u_x=v_y;  =2 if
% fn fails u_y=-v_x; and  =3 if fn fails both.
% AJR, 18 Jul 2026
% Input:    fn = string name or function handle of function
%           z = complex/real-valued point/vector/matrix 
% Output:   flag = 0 if analytic ar z, otherwise 1,2,3
[m,n] = size(z);
Z = AutoDiff( [real(z),imag(z)] );
Z = Z(:,1:end/2)+1i*Z(:,end/2+1:end);
Fval = feval(fn,Z);
U = real(Fval); V=imag(Fval);
Du = reshape(full(getderivs(U)),[],m*n,2);
Dv = reshape(full(getderivs(V)),[],m*n,2);
uxMvy=Du(:,:,1)-Dv(:,:,2);
uyPvx=Du(:,:,2)+Dv(:,:,1);
uxMvyErr=norm(uxMvy,'fro');
uyPvxErr=norm(uyPvx,'fro');
flag = (uxMvyErr>1e-8)+2*(uyPvxErr>1e-8);
end%function nonAnalytic
