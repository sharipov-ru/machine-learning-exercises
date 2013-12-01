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
%

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

for i=1:m
  % one training example: i-th row of X. (has size 1 x 401)
  ith_x = [1, X(i, :)]

  % compute activations of the 2nd layer (hidden layer, has size 25 x 1)
  a2 = sigmoid(Theta1 * ith_x')

  % add a bias unit to 2nd layer (now a2 has size 26 x 1)
  a2 = [1; a2]

  % compute activations of the 3rd layer (output layer, has size 10 x 1)
  a3 = sigmoid(Theta2 * a2)

  % choose max value from computed predictions in output layer
  [max_value, index_of_max_value] = max(a3)

  p(i) = index_of_max_value
end

% =========================================================================


end
