function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

%%% cost function %%%

% cs = 0; % cs = cost sum
% for ci = 1:m
% 	cs += -y(ci)*log(sigmoid(X(ci,:)*theta)) - (1-y(ci))*log(1-sigmoid(X(ci,:)*theta));
% end
% rs = 0; % cr = regularization sum
% for ri = 2:length(theta) % start at 2, because we don't regularize theta0
% 	rs += theta(ri)^2;
% end
% J = 1 / m * cs + lambda / (2*m) * rs;

% vectorized
sigmat = sigmoid(X*theta); % sigmoid matrix
J = (1/m)*sum((-y.*log(sigmat)) - (1-y).*log(1-sigmat)) ...% base
	+ (lambda/(2*m)) * sum(theta(2:end).^2);% regularization

%%% gradient %%%


% for feature = 1:size(X,2)
% 	s2 = 0;
% 	for i2 = 1:m
% 		s2 += (sigmoid(X(i2,:)*theta) - y(i2)) * X(i2,feature);
% 	end
% 	grad(feature) = s2 / m;
% end


% theta0
% s0 = 0; % sum for theta0
% for s0i = 1:m
% 	s0 += (sigmoid(X(s0i,:)*theta) - y(s0i)) * X(s0i,1);
% end
% grad(1) = 1/m * s0

%theta1 - theta n
% for feature = 2:length(theta)
% 	s = 0; % sum
% 	for si = 1:m
% 		s += (sigmoid(X(si,:)*theta) - y(si)) * X(si,feature);
% 	end
% 	grad(feature) = (1/m * s) + lambda/m * theta(feature); % feature is already 1 indexed
% end

% vectorized
grad(1) = (1/m) * X'(1,:) * (sigmat - y);
grad(2:end) = (1/m) * X'(2:end,:) * (sigmat - y) ... % base
			  + (lambda/m) * theta(2:end)


% =============================================================
