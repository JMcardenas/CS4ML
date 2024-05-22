%--- Description ---%
%
% Filename: compute_scalings.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: Generates and saves the m versus n scaling data for the
% polynomial regression experiments
%
% Inputs:
% poly_type - polynomials to use: either 'Legendre' or 'Hermite'
% samp_type - sampling strategy: either 'MCS' or 'CS'
% d - dimensions
% p_vals - values of hyperbolic cross index set
% num_trials - number of trials
% error_grid - error grid of size N x d
% tol - tolerance to use

function compute_scalings(poly_type,samp_type,d,p_vals,num_trials,err_grid,tol)

parfevalOnAll(@warning,0,'off','MATLAB:nearlySingularMatrix');

p_max = size(p_vals,2);
m_vals_data = zeros(num_trials,p_max);
n_vals = zeros(1,p_max);

N = size(err_grid,1);

l = 1;
for p = p_vals
    
    % generate index set
    S = generate_index_set('HC',d,p);
    n = size(S,2);
    n_vals(1,l) = n;
    
    % generate error matrices
    if isequal(poly_type,'Legendre')
        A_err_grid = generate_legendre_matrix(S,err_grid);
    else
        A_err_grid = generate_hermite_matrix(S,err_grid);
    end
    [Q,~] = qr(A_err_grid,0);
    
    % generate sampling distributions
    if isequal(samp_type,'CS')
        rng = 1:N;
        prob = zeros(N,1);
        for k = 0:d
            prob = prob+sum(abs(Q(k*N+rng,:)).^2,2);
        end
    else
        prob = ones(N,1);
    end
    prob = prob/sum(prob);
    
    % loop over trials
    parfor t = 1:num_trials
        
        m = ceil(size(S,2)/(d+1))-1;
        a = tol+1;
        
        while a > tol
            m = m+1;
            
            % generate sample points and measurement matrix
            L0 = datasample((1:N)',m,'Replace',true,'Weights',prob);
            w0 = 1./sqrt(N*prob(L0));
            w = repmat(w0,d+1,1);
            L = L0;
            for k = 1:d
                L = [L ; L0+k*N];
            end
            A = sqrt(N/m)*w.*Q(L,:);
            
            a = cond(A); % compute condition number
            
        end
        
        m_vals_data(t,l) = m;
        
        disp([samp_type,' d = ',num2str(d),' p = ',num2str(p),' n = ',num2str(n),' m = ',num2str(m),' trial = ',num2str(t)]);
        
    end
    
    l = l+1;
    
end

% set file name
file_name = [samp_type,'_d',num2str(d),'_N',num2str(N),'_trials',num2str(num_trials)];
if isequal(poly_type,'Legendre')
    file_name = ['Legendre_scaling_',file_name];
else
    file_name = ['Hermite_scaling_',file_name];
end

% save the m_values data
save(['../data/',file_name,'.mat'],'m_vals_data');

end
