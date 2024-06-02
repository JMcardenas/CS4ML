%--- Description ---%
%
% Filename: hermmat.m
% Authors: Ben Adcock, Juan M. Cardenas, Nick Dexter
% Part of the paper "A Unified Framework for Learning with Nonlinear Model 
% Classes from Arbitrary Linear Samples"
%
% Description: Generates the 1D matrix of Hermite polynomials
%
% Inputs:
% grid - a column vector of points
% k - the desired number of polynomials to use
%
% Outputs:
% A - the matrix of the first k Hermite polynomials evaluated on the grid

function A = hermmat(grid,k)

A = zeros(length(grid),k);
A(:,1) = 1;
A(:,2) = grid;
for i = 1:k-2
A(:,i+2) = grid.*A(:,i+1)/sqrt(i+1)-A(:,i)*sqrt(i/(i+1));
end

end