%--- Description ---%
%
% Filename: hermdiffmat.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: Generates the 1D matrices of Hermite polynomials and their
% first derivatives
%
% Inputs:
% grid - a column vector of points
% k - the desired number of polynomials to use
%
% Outputs:
% A - the matrix of the first k Hermite polynomials evaluated on the grid
% B - the matrix of the first derivatives of the first k Hermite 
% polynomials evaluated on the grid

function [A,B] = hermdiffmat(grid,k)

A = zeros(length(grid),k);
A(:,1) = 1;
A(:,2) = grid;
for i = 1:k-2
A(:,i+2) = grid.*A(:,i+1)/sqrt(i+1)-A(:,i)*sqrt(i/(i+1));
end

B = zeros(length(grid),k);
B(:,1) = 0;
for i=1:k-1
    B(:,i+1) = sqrt(i)*A(:,i); 
end

end