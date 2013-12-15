function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

% number of coordinates
dimension = size(x1, 1)

% distance counter
distance = 0

% iterate over coordinates and compute distance
for j=1:dimension
  distance = distance + (x1(j) - x2(j))^2
end

% gaussian kernel
sim = exp(- distance / ( 2 * sigma^2))



% =============================================================

end
