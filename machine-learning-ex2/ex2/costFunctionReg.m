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

summ = 0;
H = theta' * X';

for i = 1:m
	summ+= ((-y(i)*log(sigmoid(H(i))))-((1-y(i))*log(1-sigmoid(H(i)))));
end

summ = summ/m;
sum2 = 0;

for j = 2:length(theta)
	%Here you have to type the code for calculating the regularization term of the cost function.
	sum2 += theta(j).^2;
end

sum2 = sum2*lambda/(2*m);

J = summ + sum2;


for j = 1:length(grad)
	summ = 0;
	for i = 1:m
		summ += sum((sigmoid(H(i))-y(i))*X(i,j));
	end
	if j == 1,
		summ = summ;
	else
		summ += (lambda*theta(j));
	end;
	grad(j) = summ/m;

	
end




% =============================================================

end
