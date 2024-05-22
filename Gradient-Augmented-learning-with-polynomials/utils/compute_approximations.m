%--- Description ---%
%
% Filename: compute_approximations.m
% Authors: Anonynmous
% Part of the paper "CS4ML: A general framework for active learning with
% arbitrary data based on Christoffel functions"
%
% Description: Generates and saves the error and condition number data for 
% polynomial regression experiments
%
% Inputs:
% poly_type - polynomials to use: either 'Legendre' or 'Hermite'
% samp_type - sampling strategy: either 'MCS' or 'CS'
% fun_name - function to consider
% d - dimensions
% p_vals - values of hyperbolic cross index set
% m_vals - values of m to use
% num_trials - number of trials
% error_grid - error grid of size N x d

function compute_approximations(poly_type,samp_type,fun_name,d,p_vals,m_vals,num_trials,err_grid)

func = str2func(fun_name); % convert function name to function handle

% generate function and gradient values over the error grid
N = size(err_grid,1);
b_err_grid = func(err_grid)/sqrt(N);

% scale gradient values in Legendre case (not needed in Hermite case)
if isequal(poly_type,'Legendre')
    rng = 1:N;
    for k = 1:d
        b_err_grid(k*N+rng,1) = b_err_grid(k*N+rng,1).*sqrt(1-err_grid(rng,k).^2);
    end
end

% initialize arrays
p_max = size(p_vals,2);
err_data = zeros(num_trials,p_max);
cond_data = zeros(num_trials,p_max);

l = 1;
for p = p_vals
    
    % generate index set
    S = generate_index_set('HC',d,p);
    n = size(S,2);
    m = m_vals(l);
    
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
    for t = 1:num_trials
        
        % generate sample points and measurement matrix
        L0 = datasample((1:N)',m,'Replace',true,'Weights',prob);
        w0 = 1./sqrt(N*prob(L0));
        w = repmat(w0,d+1,1);
        L = L0;
        for k = 1:d
            L = [L ; L0+k*N];
        end
        A = sqrt(N/m)*w.*Q(L,:);
        
        % generate measurements
        y_grid = err_grid(L0,:);
        b = w.*func(y_grid)/sqrt(m);
        
        % scale gradient values in Legendre case (not needed in Hermite case)
        if isequal(poly_type,'Legendre')
            rng = 1:m;
            for k = 1:d
                b(k*m+rng,1) = b(k*m+rng,1).*sqrt(1-y_grid(rng,k).^2);
            end
        end
        
        % compute least-squares fit
        c = A\b;
        
        % compute error and condition number
        bapprox_err_grid = Q*c;
        err = norm(bapprox_err_grid - b_err_grid)/norm(b_err_grid);
        err_data(t,l) = err;
        condA = cond(A);
        cond_data(t,l) = condA;
        
        disp([samp_type,' ',fun_name,' ','d = ',num2str(d),' p = ',num2str(p),' n = ',num2str(n),' m = ',num2str(m),' trial = ',num2str(t),' err = ',num2str(err),' cond = ',num2str(condA)]);
        
    end
    
    l = l+1;
    
end

% set file name
file_name = [fun_name,'_',samp_type,'_d',num2str(d),'_N',num2str(N),'_trials',num2str(num_trials)];
if isequal(poly_type,'Legendre')
    file_name = ['Legendre_',file_name];
else
    file_name = ['Hermite_',file_name];
end

% save data
save(['../data/',file_name,'.mat'],'err_data','cond_data');

end
