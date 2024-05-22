%--- Description ---%
%
% Filename: iso_exp.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: evaluates the isotropic exponential function and its gradient
%
% Input:
% y - m x d array of sample points
%
% Output:
% b - m*(d+1) x 1 array of function and gradient values at the sample points

function b = iso_exp(y)

[m,d] = size(y);

b = zeros((d+1)*m,1);
rng = 1:m;
b(rng,1) = exp(-sum(y,2)/(2*d));
for k = 1:d
    b(k*m+rng,1) = -exp(-sum(y,2)/(2*d))/(2*d);
end

end