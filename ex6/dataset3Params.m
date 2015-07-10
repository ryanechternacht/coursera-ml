function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = .3;
% sigma = 0.1;

% COptions = [.01; .03; .1; .3; 1; 3; 10; 30];
% sigmaOptions = [.01; .03; .1; .3; 1; 3; 10; 30];

COptions = [1];
sigmaOptions = [.1];

clen = length(COptions);
slen = length(sigmaOptions);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

results = zeros(length(COptions), length(sigmaOptions));

for i=1:clen
	for j=1:slen
		model = svmTrain(X, y, COptions(i), ...
			@(x1, x2) gaussianKernel(x1, x2, sigmaOptions(j))); 
		
		predictions = svmPredict(model, Xval);

		results(i,j) = mean(double(predictions ~= yval));

		fprintf('%i %i\n', i, j);
	end
end

[~, idx] = min(results(:));
[r,c] = ind2sub(size(results), idx);

C = COptions(r);
sigma = sigmaOptions(c);

fprintf('final results: C = %f and sigma = %f\n', C, sigma);


% =========================================================================

end
