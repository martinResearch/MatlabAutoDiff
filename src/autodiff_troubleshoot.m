% Error type 1 :  assigning autodiff variables to double array parts
try
    full(AutoDiffJacobianAutoDiff(@f2_fails, 5)) % this fails
catch error
    fprintf(error.message)
end

try
    full(AutoDiffJacobianAutoDiff(@f2_fails, [5, 6])) % this fails
catch error
    fprintf(error.message)
end

%
full(AutoDiffJacobianAutoDiff(@f1_succeed, 5))
full(AutoDiffJacobianAutoDiff(@f1_succeed2, 5))

full(AutoDiffJacobianAutoDiff(@f2_succeed, 5))
full(AutoDiffJacobianAutoDiff(@f2_succeed2, 5))
full(AutoDiffJacobianAutoDiff(@f2_succeed3, 5))

function y = f1_fails(x)
y = zeros(5, 1);
y(3) = x;
end

function y = f1_succeed(x)
y = zeros(5, 1) * x; % doing a mutliplication by x to get y as autodif instance whose the drivatives matrix of the right size
y(3:4) = x;
end
function y = f1_succeed2(x)
y = zeros(5, 1, 'like', x);
y(3:4) = x;
end


function y = f2_fails(x)
y = ones(5, 1); % doing a mutliplication by x to get y as autodif instance whose the drivatives matrix of the right size
y(3:4) = x;
end

function y = f2_succeed(x)
y = ones(5, 1) + 0 * x(1); % adding 0*x(1) won't influence the result of the function when not doing AD and fixes the problem when using AD
y(3:4) = x;
end

function y = f2_succeed2(x)
y = autodiff_identity(ones(5, 1), x);
y(3:4) = x;
end

function y = f2_succeed3(x)
y = ones(5, 1, 'like', x);
y(3:4) = x;
end
