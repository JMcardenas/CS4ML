%--- Description ---%
%
% Filename: legdiffmat.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: Generates the 1D matrices of Legendre polynomials and their
% first derivatives
%
% Inputs:
% grid - a column vector of points
% k - the desired number of polynomials to use
%
% Outputs:
% A - the matrix of the first k Legendre polynomials evaluated on the grid
% B - the matrix of the first derivatives of the first k Legendre 
% polynomials evaluated on the grid

function [A,B] = legdiffmat(grid,k)

A = zeros(length(grid),k);
A(:,1) = 1;
A(:,2) = grid*sqrt(3);
for i = 2:k-1
    A(:,i+1)=(grid.* (2*i - 1).*A(:,i)./sqrt(i-1/2)- (i - 1).*A(:,i-1)./sqrt(i-3/2)).*sqrt(i+1/2)/i;
end

B = zeros(length(grid),k);
B(:,1) = 0;
B(:,2) = sqrt(3);
for i=2:k-1
    B(:,i+1)= (B(:,i-1)./sqrt(i-3/2) + (2*i-1)* A(:,i)./sqrt(i-1/2)).*sqrt(i+1/2);
end

end