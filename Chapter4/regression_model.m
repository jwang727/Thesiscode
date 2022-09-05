% Regression model
% g(r) = \sum_j z_j phi_j(r)
% where phi_j are "top hat" basis functions
%
% Prior
% z ~ N(0,Gam) conditional on g monotone and 0 <= g <= 1
% Gam is automatically specified
%
% Data
% gval(i) = g(rval1(i))
% dgval(i) = dg/dr(rval2(i))
%
% Output
% n_sampl draws from xi | data, g monotone and bounded
%
% Based on the paper "Finite-dimensional Gaussian approximation with linear
% inequality constraints"

function [r_grid,g_sampl] = regression_model(rval1,gval,rval2,dgval,varargin)

rng(30)

n_sampl = 200; % number of samples
monitor = false; % plot intermediate output

% Formatting
if size(rval1,1) == 1; rval1 = rval1'; end % want column vectors
if size(rval2,1) == 1; rval2 = rval2'; end
if size(gval,1) == 1; gval = gval'; end
if size(dgval,1) == 1; dgval = dgval'; end

% Size of the dataset
if nargin == 6
    rmin = varargin{1}; rmax = varargin{2};
else
    if ~isempty(rval1); min1 = min(rval1); else; min1 = inf; end
    if ~isempty(rval1); max1 = max(rval1); else; max1 = -inf; end
    if ~isempty(rval2); min2 = min(rval2); else; min2 = inf; end
    if ~isempty(rval2); max2 = max(rval2); else; max2 = -inf; end
    rmin = min(min1,min2); % minimum value of r in the data
    rmax = max(max1,max2); % maximum value of r in the data
end
n_data1 = length(gval); % number of observations of g
n_data2 = length(dgval); % number of observations of dg/dr

%%%%% NEED TO CHECK rval2 and rval2 ARE DISTINCT

% Set centres of basis functions in between the data
rvals = [rval1; rval2]; % locations at which we have data
factor = 8; % number of times more basis functions than data points
R = intersperse(rvals,factor)'; % centres for the basis functions
hl = diff(R); hl = [hl(1), hl];
hr = diff(R); hr = [hr, hr(end)];
N = length(R); % number of basis functions

% Basis functions
phi = @(j,r) (r > R(j)-hl(j)) .* (r < R(j)) .* (r - R(j)+hl(j)) / hl(j) ...
              - (r < R(j)+hr(j)) .* (r >= R(j)) .* (r - R(j)-hr(j)) / hr(j); % form of the basis function 
dphi = @(j,r) (r > R(j)-hl(j)) .* (r < R(j)) / hl(j) ...
              - (r < R(j)+hr(j)) .* (r >= R(j)) / hr(j); % gradient of the basis function         

% Plot basis functions
if monitor          
    figure()
    subplot(1,2,1); hold on;
    for j = 1:N
        phi_j = @(r) phi(j,r);
        fplot(phi_j,[rmin,rmax])
    end
    xlabel('r')
    ylabel("\phi_j(r)")
    subplot(1,2,2); hold on;
    for j = 1:N
        dphi_j = @(r) dphi(j,r);
        fplot(dphi_j,[rmin,rmax])
    end
    xlabel('r')
    ylabel("\phi_j'(r)")
end
    
% Plot settings
n_grid = 50; % resolution for plotting
r_grid = linspace(rmin,rmax,n_grid)'; % grid for plotting          
          
% Design matrix
Phi1 = zeros(length(rval1),N); % first part of design matrix
for i = 1:length(rval1)
    for j = 1:N
        Phi1(i,j) = phi(j,rval1(i));
    end
end
Phi2 = zeros(length(rval2),N); % second part of design matrix
for i = 1:length(rval2)
    for j = 1:N
        Phi2(i,j) = dphi(j,rval2(i));
    end
end
Phi = [Phi1; Phi2];

% Pre-prior; i.e. the Gaussian N(0,Sig)
%Gam=diag(fliplr([1:N].^2));
%Gam=10*diag([1:N].^2)
Gam = eye(N); % encodes the intuition that |z_i| = O(1)
disp('check Gam')
disp(Gam(1,1))

% Prediction matrix
Phi_pred = zeros(length(r_grid),N); 
for i = 1:length(r_grid)
    for j = 1:N
        Phi_pred(i,j) = phi(j,r_grid(i));
    end
end

% Conditional for z
y = [gval; dgval]; % data
mu = Gam * (Phi') * ((Phi * Gam * (Phi')) \ y);
Sig = Gam - Gam * (Phi') * ((Phi * Gam * (Phi')) \ (Phi * Gam));

rho = N - n_data1 - n_data2; % remaining degrees of freedom

% Cholesky decomposition
[U,L2,~] = svd(Sig);
M = U * sqrt(L2); M = M(:,1:rho);

% Unconstrained samples
if monitor
    figure(); hold on;
    z_sampl = repmat(mu,1,n_sampl) + M * randn(rho,n_sampl);
    curves = Phi_pred * z_sampl;
    plot(r_grid,curves)
    xlabel('r')
    ylabel('g(r)')
    title('Unconstrained')
end

% Encoding the boundedness and monotonicity constraint
F = zeros(N+1,rho);
g = zeros(N+1,1);
eps = zeros(N+1,1); % epsilon relaxation of the constraint
for i = 1:N+1
    if i == 1
        F(1,:) = M(factor+1,:);
        g(1) = mu(factor+1);
        eps(1) = 0; % epsilon relaxation of g >= 0 constraint
    elseif (i > 1) && (i <= N)
        F(i,:) = M(i,:) - M(i-1,:);
        g(i) = mu(i) - mu(i-1);
        eps(i) = 0.0000001; % epsilon relaxation of g' >= 0 constraint
    elseif i == (N+1)
        F(N+1,:) = - M(N-factor+1,:);
        g(N+1) = 1 - mu(N-factor+1);
        eps(i) = 0.01; % epsilon relaxation of g <= 1 constraint
    end
end

% Find any solution in the feasible set
potential = zeros(rho,1);

[z_til_0,~,flag] = linprog(potential,-F,g);
if flag == -2
    z_til_0=randn(rho);
    error("Feasible set is empty - not enough basis functions used.")
end

% Sample from (epsilon-relaxed) conditional
cov = true;
mu_til = zeros(rho,1); % mean of tilde{z}
Sig_til = eye(rho); % covariance of tilde{z}
z_til_sampl = HMC_exact(F,g+eps,Sig_til,mu_til,cov,n_sampl+1,z_til_0);
if isnan(z_til_sampl)
    error("HMC failed")
end
z_til_sampl = z_til_sampl(:,2:end); % discard initial state

% Return the output
z_sampl = repmat(mu,1,n_sampl) + M * z_til_sampl;

g_sampl = Phi_pred * z_sampl;

% Plot result
if monitor
    figure()
    plot(r_grid,g_sampl)
    xlabel('r');
    ylabel('g');
    title('posterior samples')
end

end





