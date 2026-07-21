function [flags] = nonAnalyticCheck(fn)
% classifies whether a coded function fn is analytic or not
% at random real/complex vector/matrix.  Uses nonAnalytic()
% AJR, 17 Jul 2026
% Input:    fn = string name of function
% Output:   flags = 0 if analytic ar z, otherwise 1,2,3
m = randi([2 4]);
n = randi([3 5]);
flag = nonAnalytic(fn,randn(m,1));
flags = flag;
if flag, disp([fn '(real-vector)  not analytic ' num2str(flag)]), end
flag = nonAnalytic(fn,randn(m,1)+1i*randn(m,1));
flags = [flags flag];
if flag, disp([fn '(cmplx-vector) not analytic ' num2str(flag)]), end
flag = nonAnalytic(fn,randn(m,n));
flags = [flags flag];
if flag, disp([fn '(real-matrix)  not analytic ' num2str(flag)]), end
flag = nonAnalytic(fn,randn(m,n)+1i*randn(m,n));
flags = [flags flag];
if flag, disp([fn '(cmplx-matrix) not analytic ' num2str(flag)]), end
end%function nonAnalyticCheck
