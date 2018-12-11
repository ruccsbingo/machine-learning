function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

alpha = 0.005;
i = 0;
theta0 = theta(1);
theta1 = theta(2);
theta2 = theta(3);

while (i < 1000000) 
	theta0 = theta0 - alpha * 1 / m * ((transpose(sigmoid(X * theta) - y)) * X(:, 1));
	theta1 = theta1 - alpha * 1 / m * ((transpose(sigmoid(X * theta) - y)) * X(:, 2));
	theta2 = theta2 - alpha * 1 / m * ((transpose(sigmoid(X * theta) - y)) * X(:, 3)); 

	theta = [theta0; theta1; theta2];
	i = i + 1;
endwhile

grad = theta
J = (1 / m) * (sum(-y .* log(sigmoid(X * theta)) - (1 - y) .* log(1 - sigmoid(X * theta))))

% =============================================================

end
