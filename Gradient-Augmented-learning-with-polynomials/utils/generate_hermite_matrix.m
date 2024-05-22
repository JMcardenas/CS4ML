%--- Description ---%
%
% Filename: generate_hermite_matrix.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: generates a (gradient-augmented) measurement matrix using
% tensor Hermite polynomials from an arbitrary multi-index set and collection
% of sample points
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

A = zeros((d+1)*m,n); % initialize A
pmax = max(I(:)); % find maximum polynomial degree

for i = 1:m
    y = y_grid(i,:); % select ith sample point
    
    [S,T] = hermdiffmat(y',pmax+1);
    
    % build A0
    for j = 1:n
        Sij = zeros(d,1);
        for l = 1:d
            Sij(l,1) = S(l,I(l,j)+1);
        end
        A(i,j) = prod(Sij);
    end
    
    % build Ak, k = 1,...d
    for k = 1:d 
        for j = 1:n
            Sij = zeros(d,1);
            for l = 1:d
                Sij(l,1) = S(l,I(l,j)+1);
            end
            Sij(k,1) = T(k,I(k,j)+1);
            A(i+k*m,j) = prod(Sij);
        end
    end 
end

% normalize A
for j = 1:n
    nu = I(:,j);
    b = sqrt(m*(1+sum(nu)));
    A(:,j) = A(:,j)/b;
end

end
