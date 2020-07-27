function exampleSVM()
% create some fake data
n = 30;
ndim = 3;
rng(3);
x = rand(n, ndim);

y = 2 * (rand(n, 1) > 0.5) - 1;
l2_regularization = 1e-4;

% define the loss function
    function loss = loss_fn(weights_and_bias)
        weights = weights_and_bias(1:end-1)';
        bias = weights_and_bias(end);
        margin = y .* (x * weights + bias);
        loss = max(0, 1-margin).^2;
        l2_cost = 0.5 * l2_regularization * weights' * weights;
        loss = mean(loss) + l2_cost;
    end


w_0 = zeros(1, ndim);
b_0 = 0;
weights_and_bias = [w_0, b_0];

fprintf('intial loss %f\n', loss_fn(weights_and_bias))

% heavyball gradient descent
speed = zeros(numel(weights_and_bias), 1);
tic
nbIter = 100;
Err = zeros(1, nbIter);
for k = 1:nbIter
    [J, f] = AutoDiffJacobian(@loss_fn, weights_and_bias);
    Err(k) = f;
    speed = 0.90 * speed - 0.1 * J';
    weights_and_bias(:) = weights_and_bias(:) + 0.5 * speed;
end
toc
fprintf('final loss %f\n', loss_fn(weights_and_bias))

figure(1)
plot(Err)

end
