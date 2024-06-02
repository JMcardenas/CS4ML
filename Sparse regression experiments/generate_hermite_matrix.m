%--- Description ---%
%
% Filename: generate_hermite_matrix.m
% Authors: Ben Adcock, Juan M. Cardenas, Nick Dexter
% Part of the paper "A Unified Framework for Learning with Nonlinear Model 
% Classes from Arbitrary Linear Samples"
%
% Description: generates a measurement matrix using tensor Hermite 
% polynomials from an arbitrary multi-index set and collection of sample 
% points
%
% Inputs:
% I - d x N array of multi-indices
% y_grid - m x d array of sample points
%
% Output:
% A - normalized measurement matrix

function A = generate_hermite_matrix(I,y_grid)

[d,n] = size(I); % get N (number of matrix columns) and d (dimension)
m = size(y_grid,1); % get m (number of matrix rows)

A = zeros(m,n); % initialize A
pmax = max(I(:)); % find maximum polynomial degree

for i = 1:m
    y = y_grid(i,:); % select ith sample point
    
    S = hermmat(y',pmax+1);
    
    % build A
    for j = 1:n
        Sij = zeros(d,1);
        for l = 1:d
            Sij(l,1) = S(l,I(l,j)+1);
        end
        A(i,j) = prod(Sij);
    end
end

% normalize A
A = A/sqrt(m);

end
