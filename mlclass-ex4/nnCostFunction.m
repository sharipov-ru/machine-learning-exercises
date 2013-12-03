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


triangle_1 = zeros(hidden_layer_size, input_layer_size + 1)
triangle_2 = zeros(num_labels, hidden_layer_size + 1)
for i=1:m
  % one training example: i-th row of X. (has size 1 x 401)
  a1 = [1; X(i, :)']

  % compute activations of the 2nd layer (hidden layer, has size 25 x 1)
  z2 = Theta1 * a1
  a2 = sigmoid(z2)

  % add a bias unit to 2nd layer (now a2 has size 26 x 1)
  a2 = [1; a2]

  % compute activations of the 3rd layer (output layer, has size 10 x 1)
  z3 = Theta2 * a2
  a3 = sigmoid(z3)

  y_kth = zeros(num_labels, 1)'
  y_kth(y(i)) = 1

  J = J + sum(-y_kth * log(a3) - (1 - y_kth)*log(1-a3))

  % calculate deltas (DO NOT calculate delta_1)
  delta3_kth = a3 - y_kth'
  delta2_kth = (Theta2' * delta3_kth).*sigmoidGradient([1; z2])

  % accumulate the gradient
  triangle_1 = triangle_1 + delta2_kth(2:end) * a1'
  triangle_2 = triangle_2 + delta3_kth * a2'
end

penalizedTheta1 = Theta1(:, 2:end)
penalizedTheta2 = Theta2(:, 2:end)

value = sum(sum(penalizedTheta1.^2)) + sum(sum(penalizedTheta2.^2))
reqularization = lambda / (2 * m) * value
J = 1/m * J + reqularization

% compute regularization for all Theta-jth except first column
Theta1_regularization = lambda / m * Theta1(:, 2:end)
Theta2_regularization = lambda / m * Theta2(:, 2:end)

Theta1_empty_column = zeros(size(Theta1, 1), 1)
Theta2_empty_column = zeros(size(Theta2, 1), 1)

% add regularization to penalized Theta(i,j) values
% Use empty column of zeros to not regularize first Theta-lth columns
Theta1_grad = triangle_1 / m + [ Theta1_empty_column, Theta1_regularization ]
Theta2_grad = triangle_2 / m + [ Theta1_empty_column, Theta2_regularization ]


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
