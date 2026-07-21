isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

% AJR: specify random seed to random or toreproduce
if 1, ADseed = floor(1e5*rand(1));
else  ADseed = 1;  end
if isOctave, rand ("seed", ADseed), else rng(ADseed), end

if ~exist('pagemtimes')
    addpath("./backports")    
end

% AJR, fixed June 2026
f = @(x) diff(x, 1, 3);
CheckAutoDiffJacobian(f, randn(2,3,4), 1e-8);
f = @(x) diff(x, 1, 2);
CheckAutoDiffJacobian(f, randn(3)+1i*rand(3), 1e-8);

% AJR, added July 2026
f = @(x) abs(x);
CheckAutoDiffJacobian(f, randn(2)+i*randn(2), 1e-8);

% AJR, added July 2026
f = @(x) sign(x);
z = randn(2)+i*randn(2); z=z+0.2*sign(z); % avoid singularity at 0
CheckAutoDiffJacobian(f, z, 1e-8);

f = @(x) real(x);
CheckAutoDiffJacobian(f, randn(2)+i*randn(2), 1e-8);

f = @(x) imag(x);
CheckAutoDiffJacobian(f, randn(2)+i*randn(2), 1e-8);

f = @(x) i*x;
CheckAutoDiffJacobian(f, randn(2)+i*randn(2), 1e-7);

f = @(x) real(i*x);
CheckAutoDiffJacobian(f, randn(2)+i*randn(2), 1e-7);

f = @(x) sum(x,3);
CheckAutoDiffJacobian(f, randn(2,3,5), 1e-8);
CheckAutoDiffJacobian(f, randn(2,3,5)+1i*randn(2,3,5), 1e-8);

f = @(x) mean(x,2);
CheckAutoDiffJacobian(f, randn(2,3), 1e-8);
CheckAutoDiffJacobian(f, randn(2,3)+1i*randn(2,3), 1e-8);

f = @(x) cat(2, [], x);
CheckAutoDiffJacobian(f, randn(3,3), 1e-8);

f = @(x) cat(3, [], x);
CheckAutoDiffJacobian(f, randn(3,3), 1e-8);

f = @(x) cat(3, [],[], x);
CheckAutoDiffJacobian(f, randn(3,3), 1e-8);

f = @(x) cat(3, x, []);
CheckAutoDiffJacobian(f, randn(3,3), 1e-8);

f = @(x) cat(4, [], x);
CheckAutoDiffJacobian(f, randn(3,3), 1e-8);

f = @(x) cat(4, x,[]);
CheckAutoDiffJacobian(f, randn(3,3), 1e-8);

x = randn(3, 2, 7);
f = @(y) pagemtimes(x,y);
CheckAutoDiffJacobian(f, randn(2, 5, 1), 1e-8);

x = randn(3, 2, 1)+1i*randn(3, 2, 1);
f = @(y) pagemtimes(x,y);
CheckAutoDiffJacobian(f, randn(2, 5, 7)+1i*randn(2, 5, 7), 1e-8);

x = randn(3, 2, 1, 3);
f = @(y) pagemtimes(x,y);
CheckAutoDiffJacobian(f, randn(2, 5, 7, 1), 1e-8);

y = randn(2, 4, 1);
f = @(x) pagemtimes(x, y);
CheckAutoDiffJacobian(f, randn(3, 2, 1), 1e-8);

x = randn(3, 2, 1);
f = @(y) pagemtimes(x,y);
CheckAutoDiffJacobian(f, randn(2, 4, 1), 1e-8);

x = randn(3, 2, 1);
f = @(y) pagemtimes(x,y);
CheckAutoDiffJacobian(f, randn(2, 5, 5), 1e-8);

x = randn(3, 2, 7);
f = @(y) pagemtimes(x,y);
CheckAutoDiffJacobian(f, randn(2, 5, 7), 1e-8);

y = randn(2, 5, 7, 2);
f = @(x) pagemtimes(x, y);
CheckAutoDiffJacobian(f, randn(3, 2, 7, 2), 1e-8);

f = @(x) pagemtimes(x,x);
CheckAutoDiffJacobian(f, randn(3, 3, 5), 1e-8);


f = @(x) norm(x);
CheckAutoDiffJacobian(f, randn(3,1)+1i*randn(3,1), 1e-8);
f = @(x) norm(x,1);
CheckAutoDiffJacobian(f, [-0.2818003 ,  0.00971297, -0.00271337], 1e-8)
%CheckAutoDiffJacobian(f,rand(3,2),1e-8);  disp('norm-matrix OK') %uses svd, not yet OK


% testing repmat
f = @(x) repmat(x, [3, 2]);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);

f = @(x) repmat(x, 1, 1, 10);
CheckAutoDiffJacobian(f, ones(3,3), 1e-8);

f = @(x) x(:);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);



%testing compatible size multiplication (i.e. using broadcasting)
f = @(x) x .* [3, 4, 2];
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);
f = @(x) [3, 4, 2] .* x;
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);
f = @(x) x(1, :) .* x;
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);

% various tests
f = @(x) x';
CheckAutoDiffJacobian(f, randn(2,3), 1e-8);
CheckAutoDiffJacobian(f, rand(2,3)+1i*randn(2,3), 1e-8);

f = @(x) abs(x);
CheckAutoDiffJacobian(f, randn(2, 3), 1e-8);
CheckAutoDiffJacobian(f, rand(2,3)+1i*randn(2,3), 1e-8);

f = @(x) sqrt(x);
x = randn(2,3);    x = x+0.1*sign(x); % avoid singularity at 0
CheckAutoDiffJacobian(f, x, 1e-8);
z = randn(3)+1i*randn(3); z = z+0.1*sign(z); % avoid singularity at 0
CheckAutoDiffJacobian(f, z, 1e-8);

f = @(x) cos(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);
CheckAutoDiffJacobian(f, randn(2,3)+1i*randn(2,3), 1e-7);

f = @(x) sin(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);
CheckAutoDiffJacobian(f, randn(2,3)+1i*randn(2,3), 1e-7);

f = @(x) tan(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);
CheckAutoDiffJacobian(f, randn(2,3)+1i*randn(2,3), 1e-7);

% AJR avoid magnified errors in steep gradients
f = @(x) acos(x);
CheckAutoDiffJacobian(f, rand(2, 3)-0.3, 1e-8);
CheckAutoDiffJacobian(f, randn(2, 3)+1i*randn(2,3)-0.3, 1e-8);

f = @(x) asin(x);
CheckAutoDiffJacobian(f, rand(2, 3)-0.3, 1e-8);
CheckAutoDiffJacobian(f, randn(2, 3)+1i*randn(2,3)-0.3, 1e-8);

f = @(x) atan(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);
CheckAutoDiffJacobian(f, randn(2, 3)+1i*randn(2,3), 1e-7);

f = @(x) exp(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);
CheckAutoDiffJacobian(f, randn(2, 3)+1i*randn(2,3), 1e-8);

f = @(x) log(x);
x = randn(2,3);    x = x+0.1*sign(x); % avoid singularity at 0
CheckAutoDiffJacobian(f, x, 1e-8);
z = randn(3)+1i*randn(3); z = z+0.1*sign(z); % avoid singularity
CheckAutoDiffJacobian(f, z, 1e-8);

f = @(x) tanh(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);
CheckAutoDiffJacobian(f, randn(2, 3)+1i*randn(2,3), 1e-7);

f = @(x) conj(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);
CheckAutoDiffJacobian(f, randn(2, 3)+1i*randn(2,3), 1e-7);

t = rand(3, 3);
f = @(x) cat(1, x, x*2, t);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);

f = @(x) repmat(x, [3, 4]);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);

f = @(x) diag(x);
CheckAutoDiffJacobian(f, rand(4, 1), 1e-8);
CheckAutoDiffJacobian(f, rand(4, 4), 1e-8);

f = @(x) diff(x, 1, 2);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) diff(x, 1, 1);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);
CheckAutoDiffJacobian(f, randn(4, 3)+1i*randn(4,3), 1e-8);

f = @(x) diff(x, 1, 3);
CheckAutoDiffJacobian(f, rand(4, 3, 5, 2), 1e-8);

f = @(x) x(:, end);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);
f = @(x) x(end, :);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) x(2, :);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) max(x);
CheckAutoDiffJacobian(f, rand(4, 1), 1e-8);
CheckAutoDiffJacobian(f, rand(4, 1)+1i*rand(4,1), 1e-8);

a = rand(4, 3);
f = @(x) max(a, x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

a = rand(4, 3);
f = @(x) max(x, a);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) max(x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) max(x, -x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) min(x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) min(x);
CheckAutoDiffJacobian(f, rand(4, 1), 1e-8);

a = rand(4, 3);
f = @(x) min(a, x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

a = rand(4, 3);
f = @(x) min(x, a);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) min(x, -x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) x - x(1, 2);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) x - 3;
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) 3 - x;
CheckAutoDiffJacobian(f, rand(4, 3), 1e-8);

f = @(x) x^2;
CheckAutoDiffJacobian(f, randn(3), 1e-8);

f = @(x) x^5;
CheckAutoDiffJacobian(f, rand(3), 1e-8);

f = @(x) x.^2;
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) x *  2.5;
CheckAutoDiffJacobian(f, randn(2, 3), 1e-8);

f = @(x) 2.5 * x;
CheckAutoDiffJacobian(f, randn(2, 3), 1e-8);

a = randn(2, 3);
f = @(x) a .* x;
CheckAutoDiffJacobian(f, randn(2, 3), 1e-8);

a = randn(2, 3);
f = @(x) x .* a ;
CheckAutoDiffJacobian(f, randn(2, 3), 1e-8);

% test power
f = @(x) power(x, 2.5);
CheckAutoDiffJacobian(f, randn(2, 3), 1e-8);

f = @(x) power(3,x);
CheckAutoDiffJacobian(f, randn(2, 3), 1e-8);

a = randn(2, 3);
f = @(x) power(a,x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);

a = 1+rand(2, 3); % avoid singularity for small powers
f = @(x) power(x,a);
x = rand(2, 3);
CheckAutoDiffJacobian(f, x, 1e-8);

% AJR, 7/2026: avoid singularity for small powers
f = @(x) power(x,1+x*2); 
CheckAutoDiffJacobian(f, rand(2, 3), 1e-7);

% test matrix product
f = @(x) x * x;
CheckAutoDiffJacobian(f, randn(3, 3), 1e-7);

a = randn(3, 2);
f = @(x) x * a;
CheckAutoDiffJacobian(f, randn(3, 3), 1e-7);

a = randn(2, 3);
f = @(x) a * x ;
CheckAutoDiffJacobian(f, randn(3, 3), 1e-7);

% addition

f = @(x) x + x;
CheckAutoDiffJacobian(f, randn(3, 3), 1e-7);

a = randn(3, 3);
f = @(x) x + a;
CheckAutoDiffJacobian(f, randn(3, 3), 1e-7);

a = randn(3, 3);
f = @(x) a + x ;
CheckAutoDiffJacobian(f, randn(3, 3), 1e-7);

f = @(x) inv(x);
CheckAutoDiffJacobian(f, [[1,2,3];[3,1,2];[0,4,5]], 1e-6);
% AJR: generate random not-near-singular complex matrix
n = 5;    [U,S,V]=svd(randn(n)+1i*randn(n));
CheckAutoDiffJacobian(f, U*diag(cumsum(0.3+rand(1,n)))*V', 1e-6);


% AJR, 7/2026: avoid singular for small x22
f = @(x) x / (1+x(2, 2));
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) x / 3;
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

% AJR, 7/2026: avoid singular for small x22
f = @(x) x ./ (1+x(2, 2));
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) x ./ 3;
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);
f = @(x) 3 ./ x;
x = randn(3,2);% AJR: avoid singularity by +sign
CheckAutoDiffJacobian(f, x+sign(x), 1e-8);

f = @(x) x .* abs(x);
CheckAutoDiffJacobian(f, randn(3, 3), 1e-8);

f = @(x) x .* x(2, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) x + x(:, 1);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) reshape(x, 3, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) sort(x);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) sort(x, 1);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) sort(x, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) sort(x);
CheckAutoDiffJacobian(f, rand(3, 1), 1e-8);

f = @(x) x(3, :, :);
CheckAutoDiffJacobian(f, rand(3, 2, 4), 1e-8);

f = @(x) x(:, 2, :);
CheckAutoDiffJacobian(f, rand(3, 2, 4), 1e-8);

f = @(x) sum(x);
CheckAutoDiffJacobian(f, rand(3, 1), 1e-8)
CheckAutoDiffJacobian(f, rand(1, 3), 1e-8)

f = @(x) sum(x, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) cumsum(x, 2);
CheckAutoDiffJacobian(f, rand(3, 2, 4), 1e-8);

f = @(x) cumsum(x, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) cumsum(x);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-8);

f = @(x) cumsum(x);
CheckAutoDiffJacobian(f, rand(3), 1e-8);

f = @(x) mean(x, 2);
CheckAutoDiffJacobian(f, rand(3, 2, 4), 1e-8);

f = @(x) mean(x, 1);
CheckAutoDiffJacobian(f, rand(3, 2, 4), 1e-8);

f = @(x) mean(x) ;
CheckAutoDiffJacobian(f, randn(1, 3), 1e-8);
CheckAutoDiffJacobian(f, randn(3, 1), 1e-8);
CheckAutoDiffJacobian(f, randn(3, 2), 1e-8);

%times
f = @(x) x .* abs(x);
CheckAutoDiffJacobian(f, randn(3, 2, 4), 2e-9);
t = randn(3, 2, 4);
f = @(x) x .* t;
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-8);
f = @(x) t .* x;
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-8);



f = @(x) stack2Fn('eig',x);
%disp('AJR: checking AD of symmetric arrays')
x=randn(4); x=x+x';
CheckAutoDiffJacobian(f, x, 1e-8);
x=randn(3)+1i*randn(3); x=x+x';
CheckAutoDiffJacobian(f, x, 1e-8);
%disp('AJR: checking AD of non-symmetric arrays')
CheckAutoDiffJacobian(f, randn(4), 1e-7);
CheckAutoDiffJacobian(f, randn(3)+1i*randn(3), 1e-7);


f = @(x) x';
CheckAutoDiffJacobian(f, randn(3, 3), 1e-8);

f = @(x) permute(x, [3, 1, 2]);
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-8);

f = @(x) x - 1;
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-8);

f = @(x) x + 1;
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-8);

f = @(x) [x, x * 2];
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-8);

f = @(x) [x; x * 2];
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-8);

f = @(x) det(x);
CheckAutoDiffJacobian(f, randn(2, 2), 1e-8);

%AJR, 7/2026: include complex value test
f = @(x) det(x);
CheckAutoDiffJacobian(f, 2*randn(3)+1i*randn(3), 1e-7 );

f = @(x) det(x);
CheckAutoDiffJacobian(f, randn(4, 4), 1e-8);

f = @(x) sinh(x);
CheckAutoDiffJacobian(f, randn(4, 4), 1e-8);

f = @(x) cosh(x);
CheckAutoDiffJacobian(f, randn(4, 4), 1e-8);

f = @(x) asinh(x);
CheckAutoDiffJacobian(f, randn(4, 4), 1e-8);
z = 2*randn(3)+1i*randn(3); %AJR, 7/2026: include complex value test
CheckAutoDiffJacobian(f, z, 1e-8 );

% AJR, 7/2026: the following used to lack testing the real cases of x>1
% also adjust tolerance near the sqrts at x=+-1
f = @(x) acosh(x);
z = 2*randn(3)+1i*randn(3);
CheckAutoDiffJacobian(f, z, 1e-8/min(abs(z(:).^2-1)) );
x = randn(3);
CheckAutoDiffJacobian(f, x, 1e-8/min(abs(x(:).^2-1)) );

% AJR avoid magnified errors in steep gradients
f = @(x) atanh(x);
CheckAutoDiffJacobian(f, rand(4, 4)-0.3, 1e-8);

% some other tests

n = 300;
A = sprand(n, n, 0.2);
x0 = rand(n, 1);
f = @(x) diag(x.^2) * (A * x);
CheckAutoDiffJacobian(f, x0, 1e-8);
f = @(x) (x.^2) .* (A * x);
CheckAutoDiffJacobian(f, x0, 1e-8);
if ~isOctave
    f = @(x) ((x.^2) .* A) * x;
end
CheckAutoDiffJacobian(f, x0, 1e-8);


f = @(x) fft(x);
dim = randi([3 9],1,2);
X = randn(dim)+1i*randn(dim);
CheckAutoDiffJacobian(f, X, 1e-8);

f = @(x) ifft(x);
dim = randi([3 9],1,2);
X = randn(dim)+1i*randn(dim);
CheckAutoDiffJacobian(f, X, 1e-8);

% AJR: uses extra function to stack result U,S,V
% and adjust tolerance when close singular values
f = @(x) stack3Fn('svd',x);
X = randn(randi([3 9],1,2));
CheckAutoDiffJacobian(f, X, 1e-8*(1+1/min(abs(diff(svd(X))))) );
disp('AD.svd  OK for real')
dim = randi([3 9],1,2);
X = randn(dim)+1i*randn(dim);
CheckAutoDiffJacobian(f, X, 1e-8*(1+1/min(abs(diff(svd(X))))) );
disp('AD.svd  OK for cmplx')



disp('**** Now check on Analytic/holomorphic/...')
analyticFns={'transpose','sin','cos','tan','exp','log','sqrt','diff','sum','mean','cumsum','sinh','cosh','tanh','asin','acos','atan','asinh','acosh','atanh'}
for fn=analyticFns,  nonAnalyticCheck(fn{1});  end;

fn = 'inv', n=randi([3 5]);
if nonAnalytic(fn,randn(n))
    disp([fn '(real-matrix)  not analytic']), end
if nonAnalytic(fn,randn(n)+1i*randn(n))
    disp([fn '(cmplx-matrix)  not analytic']), end

fn = @(X) stack2Fn('eig',X) 
X=randn(randi([3 5])); X=X+X';
if nonAnalytic(fn,X)
    disp(['eig(sym-real-matrix)  not analytic']), end
n=randi([2 4]); X=randn(n)+1i*randn(n); X=X+X';
if nonAnalytic(fn,X)
    disp(['eig(sym-cmplx-matrix) not analytic']), end
X=randn(randi([3 5]));
if nonAnalytic(fn,X)
    disp(['eig(gen-real-matrix)  not analytic']), end
n=randi([2 4]); X=randn(n)+1i*randn(n);
if nonAnalytic(fn,X)
    disp(['eig(gen-cmplx-matrix) not analytic']), end

disp('** Reaffirm non-analytic functions follow')
for fn={'imag','real','conj','abs','sign','norm','ctranspose'}
    nonAnalyticCheck(fn{1});
end;



function ABC = stack3Fn(fn,X)
    % AJR, 18 Jul 2026
    [m,n] = size(X); 
    [A,B,C] = feval(fn,X);
    if strcmp(fn,'svd')  % return only k columns, and
        % ensure col.max-V is real-positive to be consistent with AD.svd()
        k = 1:min(m,n);
        [~,j] = max(abs(C(:,k))); 
        rot = conj( sign(C(n*(k-1)+j)) );
        ABC = [ A(:,k).*rot; B(k,k); C(:,k).*rot ];
    else ABC = [A;B;C];
    end;%if svd
end%function stack3Fn

function VD = stack2Fn(fn,X)
    % AJR, 18 Jul 2026
    [V,D] = feval(fn,X);
    % always need to sort in a consistent order whether real
    % or complex --- especially for cases when all real
    % e-vals are perturbed to complex 
    d = diag(D);
    [d,j] = sort(d,'ComparisonMethod','real');
    D = diag(d); V = V(:,j);
    % ensure consistent normalised e-vecs, make max-abs element =1
    [~,iM]=max(abs(V));
    for j=1:size(V,2),  V(:,j) = V(:,j)/V(iM(j),j);  end
    VD = [V D];
end%function stack2Fn

