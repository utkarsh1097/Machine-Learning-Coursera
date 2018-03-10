function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
  delta = zeros(size(X, 2), 1); %size(X, 2) gives the column dimension of matrix X, which is 2. This is more general approach
  for i = 1:m
    delta = delta+(X(i,:)*theta-y(i, 1))*X(i,:)'; %Using vectorized form of gradient descent
                                                  %%y(i, 1) and y(i) is same for column vector. Similarly for row vector
  endfor
  delta = delta*(alpha/m);
  theta = theta - delta;


    % ============================================================

    % Save the cost J in every iteration    
  J_history(iter) = computeCost(X, y, theta);

endfor

end
