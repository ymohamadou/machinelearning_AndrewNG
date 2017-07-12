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

% Compute the sigmoid 
z = X*theta;
h = sigmoid(z);

%Compute the cost function
bias = 0;
for iter=2:length(theta)
  bias = bias + theta(iter)^2;
end
  
J = (-y'*log(h)-(1-y)'*log(1-h)) / m + (lambda*bias)/(2*m);

%compute de gradient
grad = (X'*(h-y))/m;
grad(2:end, 1) = grad(2:end, 1) + (lambda*theta(2:end, 1))/m;
% =============================================================

end
