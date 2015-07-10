function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X = mxn
% y = mx1
% theta = nx1

% from week 1
% predictions = X * theta;
% squaredError = (predictions - y).^2;
% J = 1 / (2*m) * sum(squaredError);


J = 1/(2*m) * sum((X*theta - y).^2) + ...
	lambda/(2*m) * sum(theta(2:end).^2);


% from vectorized logistic regression
% grad(1) = (1/m) * X'(1,:) * (sigmat - y);
% grad(2:end) = (1/m) * X'(2:end,:) * (sigmat - y) ... % base
% 			  + (lambda/m) * theta(2:end)


grad(1) = 1/m * X(:,1)' * (X*theta - y); % 1xm * (mxn * nx1) = 1
grad(2:end) = 1/m * X(:,2:end)' * (X*theta - y) + ... % (n-1)xm * (mxn * nx1) = (n-1)x1
			  lambda/m * theta(2:end); % (n-1)x1




% =========================================================================

grad = grad(:);

end
