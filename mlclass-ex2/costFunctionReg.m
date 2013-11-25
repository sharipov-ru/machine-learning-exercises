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

% select all theta values
penalizedParameters = theta

% remove first theta-zero value from selection
penalizedParameters(1, :) = []

% compute regularization term
regularizationTerm = lambda / (2 * m) * sum(penalizedParameters.^2)

% compute cost function with regularization term
J = 1 / m * sum(-y' * log(sigmoid(X * theta)) - (1-y')*(log(1-sigmoid(X * theta)))) + regularizationTerm


unRegularizedGradients = 1 / m * X' * (sigmoid(X * theta) - y)

% calculate 0-th gradient
grad(1) = unRegularizedGradients(1)

% calculate other gradients
for j=2:size(grad)
  grad(j) = unRegularizedGradients(j) + lambda / m * penalizedParameters(j-1);
end;

% =============================================================

end
