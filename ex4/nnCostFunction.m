function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% make vectorized Y
I = eye(num_labels);
Y = zeros(m, num_labels); % vectorized version of our output
for i = 1:m
	Y(i,:) = I(y(i), :); % grab the correct row from the identity matrix
end

% calculate hx (from ex3)
a1 = [ones(m,1) X]'; % (n+1) x m
z2 = Theta1 * a1; % hls x (n+1) * (n+1) x m = hls x m
a2 = [ones(1, size(z2,2)); sigmoid(z2)]; % (hls+1) x m
z3 = Theta2 * a2; % k x (hls+1) * (hls+1) x m = k x m
a3 = sigmoid(z3); % k x m
hx = a3';

% cost (loop based)
% costsum = 0;
% for i = 1:m
% 	for k = 1:num_labels
% 		costsum += -Y(i,k) * log(hx(i,k)) - (1-Y(i,k)) * (log(1 - hx(i,k)));
% 	end
% end

% vectorized cost
J = 1/m * sum(sum( (-Y) .* log(hx) - (1-Y) .* log(1 - hx)));

% regularization penalty (regpen)
t1 = Theta1(:,2:end); % remove bias feature for regularization
t2 = Theta2(:,2:end); % ditto
regpen = lambda / (2*m) * (sum(sum(t1.^2)) + sum(sum(t2.^2)));

J += regpen;

D1 = zeros(size(Theta1_grad));
D2 = zeros(size(Theta2_grad));

% back propogation
for t = 1:m % loop through each training example
	a3t = a3(:, t); % this training examples a3 column vector
	d3 = a3t - Y(t,:)'; % (k x 1)

	% d2 = (Theta2 * d3) .* g'(z2) (remove 0 term)
	a2t = a2(:, t); % training examples a2 column vector (hls+1 x 1)
	z2t = z2(:,t); % training examples z2 column vector (hls x 1)
	gderiv = a2t .* (1 - a2t); % hls+1 x 1
	d2 = (Theta2' * d3) .* gderiv; % (hls+1 x k) * (kx1) .* (hls+1 x 1) = (hls+1 x 1)

	d2_nobias = d2(2:end); % remove bias

	% D(l) := D(l) + d(l+1) * a(l)'
	a1t = a1(:, t); % this training examples a1 column vector

	D1 += d2_nobias * a1t'; % (hls x 1) * (1 x n+1) = (hls x n+1)
	D2 += d3 * a2t'; % (kx1) * (1 x hls+1) = k x hls+1
end

Theta1_grad = (1/m) * D1;
Theta2_grad = (1/m) * D2;

% add regularization
Theta1_grad(:, 2:end) += lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) += lambda / m * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
