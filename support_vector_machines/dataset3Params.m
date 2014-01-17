function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
n = 8;
cross_error_min = 1e99;
for i = 0:n-1
	for j = 0:n-1
		C_iter = 0.01*3^i;
		sigma_iter = 0.01*3^j;
		model = svmTrain(X, y, C_iter, @(x1, x2) gaussianKernel(x1, x2, sigma_iter));
		y_predict = svmPredict(model, Xval);
		cross_error = sum((yval - y_predict).^2) / length(yval) / 2;
		if cross_error < cross_error_min
			cross_error_min = cross_error;
			C = C_iter
			sigma = sigma_iter
	end
end







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







% =========================================================================

end
