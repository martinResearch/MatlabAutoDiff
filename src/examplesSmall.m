% Examples:
%

%%  Simple example
%
f = @(x) [sin(x(1)), cos(x(2)), tan(x(1) * x(2))];
JAD = full(AutoDiffJacobianAutoDiff(f, [1, 1]))
JFD = AutoDiffJacobianFiniteDiff(f, [1, 1])
JAD + i * JFD % comparing visualy using complex numbers
fprintf('max abs difference = %e\n', full(max(abs(JAD(:) - JFD(:)))));

%% Speedup illustration

f = @(x) (log(x(1:end - 1)) - tan(x(2:end)));
tic;
JAD = AutoDiffJacobianAutoDiff(f, 0.5*ones(1, 5000));
timeAD = toc;
tic;
JFD = sparse(AutoDiffJacobianFiniteDiff(f, 0.5 * ones(1, 5000)));
timeFD = toc;
fprintf('speedup AD vs FD = %2.1f\n', timeFD/timeAD);
fprintf('max abs difference = %e\n', full(max(abs(JAD(:) - JFD(:)))));

%%  N-D array support

f = @(x) sum(x.^2, 3);
AutoDiffJacobianFiniteDiff(f, ones(2, 2, 2))

%%
A = rand(4, 3);
b = rand(4, 1);
f = @(x) A \ x


AutoDiffJacobianFiniteDiff(f, b) + i * full(AutoDiffJacobianAutoDiff(f, b))

%%
A = rand(4, 3);
b = rand(4, 2);
f = @(x) A \ x

AutoDiffJacobianFiniteDiff(f, b) + i * full(AutoDiffJacobianAutoDiff(f, b))

%%
A = rand(3, 4);
f = @(x) x(:, 1:3) \ x(:, 4)
AutoDiffJacobianFiniteDiff(f, A) + i * full(AutoDiffJacobianAutoDiff(f, A))

%%
A = rand(2, 2);
A = 0.5 * (A + A');

AutoDiffJacobianFiniteDiff(@(x) (eig(x)), A) + i * full(AutoDiffJacobianAutoDiff(@(x) (eig(x)), A))

%%

AutoDiffJacobianFiniteDiff(@(x) sort(eig(x)), A) + i * full(AutoDiffJacobianAutoDiff(@(x) sort(eig(x)), A))

%%

f = @(x) selectKthOutput(2, 1, @eig, 0.5*(x + x'));

AutoDiffJacobianFiniteDiff(f, A) + i * full(AutoDiffJacobianAutoDiff(f, A))

%%
f = @(x) sort(diag(selectKthOutput(2, 2, @eig, x)));
AutoDiffJacobianFiniteDiff(f, A) + i * full(AutoDiffJacobianAutoDiff(f, A))
