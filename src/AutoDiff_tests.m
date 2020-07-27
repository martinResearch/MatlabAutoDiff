rng(1)


f = @(x) norm(x);
CheckAutoDiffJacobian(f, rand(1, 3), 1e-9);
%CheckAutoDiffJacobian(f,rand(3,2),1e-9); uses svd, not coded yet


% testing repmat
f = @(x) repmat(x, [3, 2]);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) x(:);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

%testing compatible size multiplication (i.e. using broadcasting)
f = @(x) x .* [3, 4, 2];
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);
f = @(x) [3, 4, 2] .* x;
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);
f = @(x) x(1, :) .* x;
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

% various tests
f = @(x) x';
CheckAutoDiffJacobian(f, randn(2, 3), 1e-9);

f = @(x) abs(x);
CheckAutoDiffJacobian(f, randn(2, 3), 1e-9);

f = @(x) sqrt(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-8);

f = @(x) cos(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) sin(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) tan(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) acos(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) asin(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) atan(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) exp(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) log(x);
CheckAutoDiffJacobian(f, rand(2, 3)+0.1, 1e-9);

f = @(x) tanh(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);


f = @(x) conj(x);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

t = rand(3, 3);
f = @(x) cat(1, x, x*2, t);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) repmat(x, [3, 4]);
CheckAutoDiffJacobian(f, rand(2, 3), 1e-9);

f = @(x) diag(x);
CheckAutoDiffJacobian(f, rand(4, 1), 1e-9);
CheckAutoDiffJacobian(f, rand(4, 4), 1e-9);


f = @(x) diff(x, 1, 2);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);
f = @(x) diff(x, 1, 1);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) x(:, end);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);
f = @(x) x(end, :);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) x(2, :);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) max(x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) max(x, -x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) min(x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) min(x, -x);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) x - x(1, 2);
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) x - 3;
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) 3 - x;
CheckAutoDiffJacobian(f, rand(4, 3), 1e-9);

f = @(x) x^2;
CheckAutoDiffJacobian(f, 3, 1e-9);

f = @(x) x^3;
CheckAutoDiffJacobian(f, rand(3, 3), 1e-9);

f = @(x) x.^2;
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);

f = @(x) inv(x);
CheckAutoDiffJacobian(f, randn(3, 3), 1e-9);

f = @(x) x / x(2, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-7);

f = @(x) x / 3;
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);

f = @(x) x ./ x(2, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);
f = @(x) x ./ 3;
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);
f = @(x) 3 ./ x;
CheckAutoDiffJacobian(f, randn(3, 2), 1e-4);

f = @(x) x .* abs(x);
CheckAutoDiffJacobian(f, randn(3, 3), 1e-9);

f = @(x) x .* x(2, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);

f = @(x) x + x(:, 1);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);

f = @(x) reshape(x, 3, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);

f = @(x) sort(x, 1);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);
f = @(x) sort(x, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);


f = @(x) x(3, :, :);
CheckAutoDiffJacobian(f, rand(3, 2, 4), 1e-9);

f = @(x) x(:, 2, :);
CheckAutoDiffJacobian(f, rand(3, 2, 4), 1e-9);

f = @(x) sum(x, 2);
CheckAutoDiffJacobian(f, rand(3, 2), 1e-9);

f = @(x) mean(x, 2);
CheckAutoDiffJacobian(f, rand(3, 2, 4), 1e-9);

f = @(x) mean(x, 1);
CheckAutoDiffJacobian(f, rand(3, 2, 4), 1e-9);

%times
f = @(x) x .* abs(x);
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-9);
t = randn(3, 2, 4);
f = @(x) x .* t;
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-9);
f = @(x) t .* x;
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-9);


f = @(x) eig(0.5*(x + x')); % we need to symetrize the matrix here so that the input to eig stays symetric when doing finite differences
t = randn(4, 4);
CheckAutoDiffJacobian(f, t, 1e-8);

f = @(x) selectKthOutput(2, 1, f, x);
t = randn(3, 3);
CheckAutoDiffJacobian(f, t+t', 1e-9);


f = @(x) x';
CheckAutoDiffJacobian(f, randn(3, 3), 1e-9);

f = @(x) permute(x, [3, 1, 2]);
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-9);

f = @(x) x - 1;
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-9);

f = @(x) x + 1;
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-9);

f = @(x) [x, x * 2];
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-9);

f = @(x) [x; x * 2];
CheckAutoDiffJacobian(f, randn(3, 2, 4), 1e-9);

f = @(x) det(x);
CheckAutoDiffJacobian(f, randn(2, 2), 1e-9);

f = @(x) det(x);
CheckAutoDiffJacobian(f, randn(2, 2), 1e-9);

% some other tests

n = 300;
A = sprand(n, n, 0.2);
x0 = rand(n, 1);
f = @(x) diag(x.^2) * (A * x);
CheckAutoDiffJacobian(f, x0, 1e-8);
f = @(x) (x.^2) .* (A * x);
CheckAutoDiffJacobian(f, x0, 1e-8);
f = @(x) ((x.^2) .* A) * x;
CheckAutoDiffJacobian(f, x0, 1e-8);




