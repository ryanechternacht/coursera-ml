function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

newTheta = zeros(length(theta),1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    for feature = 1:size(X,2)
        lss = 0;
        for i = 1:m
            lss += (X(i,:) * theta - y(i)) * X(i,feature);
        end
        newTheta(feature,1) = theta(feature,1) - alpha / m * lss;
    end 

    theta = newTheta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    %sprintf("J(theta) = %f after %d iterations", J_history(iter), iter)

end
