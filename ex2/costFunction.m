function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% s = 0;
% for i = 1:m
% 	s += -y(i)*log(sigmoid(X(i,:)*theta)) - (1-y(i))*log(1-sigmoid(X(i,:)*theta));
% end
% J = 1/m * s;

% vectorized implementation
sigmat = sigmoid(X*theta); % sigmoid matrix
J = (1/m)*sum((-y.*log(sigmat)) - (1-y).*log(1-sigmat));


for feature = 1:size(X,2)
	s2 = 0;
	for i2 = 1:m
		s2 += (sigmoid(X(i2,:)*theta) - y(i2)) * X(i2,feature);
	end
	grad(feature) = s2 / m;
end



% =============================================================

end
