function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.

% % % each row is a sample (each column is a feature) % % %

%hls = hidden layer size
% X = m x (n+1) 
% theta1 = hls x (n+1)
% theta2 = k x (hls+1)
% a1 = m x n+1
% a2 = m x hls+1
% a3 = m x k

% a1 = [ones(m, 1) X];  % m x n+1
% z2 = a1 * Theta1';    % m x hls
% a2 = sigmoid(z2);     % m x hls
% a2 = [ones(m,1) a2];  % m x hls+1
% z3 = a2 * Theta2';    % m x k
% a3 = sigmoid(z3);     % m x k

% [best_prediction, best_prediction_label] = max(a3');

% p = best_prediction_label';



% % % each column is a sample (each row is a feature) % % %

%hls = hidden layer size
% X = m x (n+1) 
% theta1 = hls x (n+1)
% theta2 = k x (hls+1)
% a1 = m x n+1
% a2 = m x hls+1
% a3 = m x k

a1 = [ones(m,1) X]'; % (n+1) x m
z2 = Theta1 * a1; % hls x (n+1) * (n+1) x m = hls x m
a2 = [ones(1, size(z2,2)); sigmoid(z2)]; % (hls+1) x m
z3 = Theta2 * a2; % k x (hls+1) * (hls+1) x m = k x m
a3 = sigmoid(z3); % k x m

[best_prediction, best_prediction_label] = max(a3); % columnwise max

p = best_prediction_label';

% =========================================================================


end
