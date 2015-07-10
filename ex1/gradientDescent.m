function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

newTheta = [0; 0];

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % lss1 = 0;
    % for i = 1:m
    %     lss1 += (X(i,:) * theta - y(i)) * X(i,1);
    % end
    % newTheta(1,1) = theta(1,1) - alpha / m * lss1;

    % lss2 = 0; 
    % for i = 1:m
    %     lss2 += (X(i,:) * theta - y(i)) * X(i,2);
    % end
    % newTheta(2,1) = theta(2,1) - alpha / m * lss2;

    % theta = newTheta;

    % vectorized of above
    theta = theta - alpha/m * (((X * theta) - y)' * X)';
    % end





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %sprintf("J(theta) = %f after %d iterations", J_history(iter), iter)
end

end
