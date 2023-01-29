# Goal

This project implements a Matlab/Octave non-intrusive forward automatic differentiation method, ([wikipedia definition here](https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_accumulation)) based on operator overloading. This does not provide backward mode or higher order derivatives. It enables precise and efficient computation of the Jacobian of a function. This contrasts with numerical differentiation (a.k.a finite differences) that is unprecise due to roundoff errors and that cannot exploit the sparsity of the derivatives.

In contrast with most existing automatic differentiation Matlab toolboxes:

 * Derivatives are represented as sparse matrices, which yield to large speedups with respect to other forward mode methods when the Jacobian of the function we aim to differentiate is sparse or when intermediate accumulated Jacobian matrices are sparse (see the image denoising example). 
 * N dimensional arrays are supported while many Matlab automatic differentiation toolboxes only support scalars, vectors and 2D matrices

It is likely that the speed could be improved by representing Jacobian matrices by their transpose, due to the way Matlab represents internally sparse matrices. The document [1] describes a method similar to the one implemented here and could be a very valuable source to improve the code.

It has been tested on Matlab 2014a and Octave 4.0.0, but the example using the anonymous function @(x)eig(x) does not work on octave as octave does not call the overloaded eig function once anonymized.

Note that backward differentation (a.k.a. gradients back-propagation in deep learning) is going to me much faster than forward differentiation when the dimension of the output is small in comparison to the dimension of the input. Forward differentiation is of interest when solving a non linear least squares for examples through Levenberg-Marquardt minimization where we want to compute the full jacobian matrix of the residuals.

# Licence
	
Free BSD License

# Examples

more examples can be found in [./src/AutoDiffExamples.m](./src/examplesSmall.m)

 * Simple example

```c

		>> f=@(x) [sin(x(1)) cos(x(2)) tan(x(1)*x(2))];
		>> JAD=full(AutoDiffJacobianAutoDiff(f,[1,1]))
		>> JFD=AutoDiffJacobianFiniteDiff(f,[1,1])
		>> JAD+i*JFD % comparing visualy using complex numbers
		>> fprintf('max abs difference = %e\n',full(max(abs(JAD(:)-JFD(:)))));
		ans =

		   0.5403 + 0.5403i   0.0000 + 0.0000i
		   0.0000 + 0.0000i  -0.8415 - 0.8415i
		   3.4255 + 3.4255i   3.4255 + 3.4255i
		
		max absolute difference = 1.047984e-10
```

	
 * Automatic Differentiation vs Finite Differences speedup illustration 
```c
		>> f=@(x) (log(x(1:end-1))-tan(x(2:end)))
		>> tic; JAD=AutoDiffJacobianAutoDiff(f,0.5*ones(1,5000));timeAD=toc;
		>> tic; JFD=sparse(AutoDiffJacobianFiniteDiff(f,0.5*ones(1,5000)));timeFD=toc;
		>> fprintf('speedup AD vs FD = %2.1f\n',timeFD/timeAD)
		>> fprintf('max abs difference = %e\n',full(max(abs(JAD(:)-JFD(:)))));

		speedup AD vs FD = 192.2
		max absolute difference = 5.351097e-11
```


 * N-D arrays support
```c
		>> f=@(x) sum(x.^2,3);
		>> AutoDiffJacobianFiniteDiff(f,ones(2,2,2))
		ans =

		    2.0000         0         0         0    2.0000         0         0         0
		         0    2.0000         0         0         0    2.0000         0         0
		         0         0    2.0000         0         0         0    2.0000         0
		         0         0         0    2.0000         0         0         0    2.0000

```
* a simple images denoising example using a total variation (TV) regularization can be found in  [./src/AutoDiffExamples.m](./src/exampleDenoise.m)
```c
		f=@(x) sum(x.^2,3);
		AutoDiffJacobianFiniteDiff(f,ones(2,2,2))
		Inoisy=zeros(100,100)*0.2;
		Inoisy(30:70,30:70)=0.8 
		Inoisy=Inoisy+randn(100,100)*0.3
		imshow(Inoisy)
		epsilon=0.0001

		%define the objective function using an anonymous function
		func=@(I) sum(reshape((Inoisy-I).^2,1,[]))+...
		sum(reshape(sqrt(epsilon+diff(I(:,1:end-1),1,1).^2+...
		diff(I(1:end-1,:),1,2).^2),1,[]))
		func(Inoisy)
		speed=zeros(numel(Inoisy),1);
		% heavy ball gradient descent, far from the best optimization method but simple
		for k=1:500
		      [J,f]=AutoDiffJacobian(func,Inoisy);
		      speed=0.95*speed-0.05*J';
		      Inoisy(:)=Inoisy(:)+0.05*speed;
		      imshow(Inoisy);
		end
		

```
* a simple SVM classifier training example can be found in  [./src/AutoDiffExamples.m](./src/exampleSVM.m)
```c
	function exampleSVM()
		% create some fake data
		n=30;
		ndim=3;
		rng(3);
		x = rand(n, ndim);	 	
		y = 2 * (rand(n,1) > 0.5) - 1;
		l2_regularization = 1e-4;
		 
		% define the loss function
		function loss=loss_fn(weights_and_bias)
		       weights=weights_and_bias(1:end-1)' ;
		       bias=weights_and_bias(end);
		       margin = y .* (x*weights + bias);
		       loss = max(0, 1 - margin) .^ 2;
		       l2_cost = 0.5 * l2_regularization * weights'* weights;
		       loss = mean(loss) + l2_cost;
		end

		w_0=zeros(1,ndim);
		b_0 = 0;
		weights_and_bias=[w_0,b_0];	 	
		fprintf('intial loss %f\n',loss_fn(weights_and_bias))

		% heavyball gradient descent
		speed=zeros(numel(weights_and_bias),1);
		tic
		nbIter=100;
		Err=zeros(1,nbIter);
		for k=1:nbIter
		    [J,f]=AutoDiffJacobian(@loss_fn,weights_and_bias);
		    Err(k)=f;
		    speed=0.90*speed-0.1*J';
		    weights_and_bias(:)=weights_and_bias(:)+0.5*speed;
		end
		toc
		fprintf('final loss %f\n',loss_fn(weights_and_bias))
		figure(1)
		plot(Err)
		end
```
# Related projects
* [Autodiff_R2016b](https://uk.mathworks.com/matlabcentral/fileexchange/61849-autodiff_r2016b) and [Autodiff_R2015b](http://mathworks.com/matlabcentral/fileexchange/56856-autodiff) by Ultrich Reif. It uses cells to represent derivatives and uses loops instead of vectorized operations in some of the functions, which may make it too slow when using large matrices.

* [TOMLAB/MAD](http://tomopt.com/tomlab/products/mad/). Not free. Method described in [1]. Like our code it uses operator overloading and can use sparse matrices to store directional derivatives.

* [Automatic Differentiation for Matlab](http://www.mathworks.com/matlabcentral/fileexchange/15235-automatic-differentiation-for-matlab/) by Martin Fink.
 Forward mode AD using operator overloading. Does not work with ND arrays. Not efficient for functions with sparse jacobians as it uses dense 3D arrays to store the derivatives.

* [Automatic Differentiation with Matlab Objects](http://mathworks.com/matlabcentral/fileexchange/26807-automatic-differentiation-with-matlab-objects) by William Mcllhagga. Supports sparse jacobians but does not support ND arrays or even some matrix operations. This will fail.
```c		
		f=@(x) sum(x'*x)
		[x,dx] = autodiff(rand(5,1),f)

```
* [madiff](https://github.com/gaika/madiff)
  Reverse mode AD using operator overloading. Operators like transpose are not coded yet  at the date of july 2016 . This will fail
```c		
		f=@(x)(sum(x'*x))
		f(rand(20,1))
		f_grad = @(x) adiff(f, x);
		f_grad(rand(20,1))

```

* [AD_deriv](https://github.com/jborggaard/AD_Deriv) by Jeff Borggaard
  works only with scalars at the date of july 2016 (no vector , matrices and NDarrays)

* [Sparsegrad](https://pypi.org/project/sparsegrad/) by Marek Szymanski. Python. Automatically and efficiently calculates analytical sparse Jacobian of arbitrary numpy vector valued functions. Does not support ND arrays yet in August 2019.

* [PTNobel/AutoDiff](https://github.com/PTNobel/AutoDiff) By Part Nobel. Python. Non-intrusive Forward differentiation with sparse Jacobians support.

## Projects that use this library

* [pde1dm](https://github.com/wgreene310/pde1dm). 1D Partial Differential Equation Solver for MATLAB and Octave.
* [NSCool_Old](https://github.com/Axect/NSCool_Old). Neutron Star Cooling simulation.

Please add a comment in [this issue](https://github.com/martinResearch/MatlabAutoDiff/issues/16) if you which to add you project in this listing. I am very interested  in knowing what it has been used for.


## References

[1] Forth, Shaun A. *An Efficient Overloaded Implementation of Forward Mode Automatic Differentiation in MATLAB*
ACM Trans. Math. Softw. 2006 [pdf](https://core.ac.uk/download/files/23/139791.pdf)
