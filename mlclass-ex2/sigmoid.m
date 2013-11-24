function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


numberRows = size(g, 1)
numberColumns = size(g, 2)

for i = 1:numberRows
  for j = 1:numberColumns
    element = z(i, j)
    g(i, j) = 1 / ( 1 + exp(-element))
  end;
end;





% =============================================================

end
