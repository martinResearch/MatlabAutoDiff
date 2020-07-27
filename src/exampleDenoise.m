Inoisy = zeros(100, 100) * 0.2;
Inoisy(30:70, 30:70) = 0.8;
Inoisy = Inoisy + randn(100, 100) * 0.3;
imshow(Inoisy)

epsilon = 0.0001;
func = @(I) sum(reshape((Inoisy-I).^2, 1, [])) + sum(reshape(sqrt(epsilon + diff(I(:, 1:end-1), 1, 1).^2 + diff(I(1:end-1, :), 1, 2).^2), 1, []));
func(Inoisy)

% heavyball gradient descent
speed = zeros(numel(Inoisy), 1);
tic
for k = 1:500
    [J, f] = AutoDiffJacobian(func, Inoisy);
    speed = 0.95 * speed - 0.05 * J';
    Inoisy(:) = Inoisy(:) + 0.05 * speed;
    imshow(Inoisy);
end
toc
