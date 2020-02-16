%INITIALIZATION
%params
%N: number of samples
%theta: linspace
%R: radius of big sphere
%r: radius of inner sphere
%n: number of spirals
%w: width factor.
N = 2000;
theta = linspace(0,2*pi,N)';
theta = theta(randperm(N),:);
R = 1;
r = .3;
n = 20;
w = .05;

x = helix(theta, R, r, n, w);
scatter3(x(:,1), x(:,2), x(:,3), n, theta, 'filled');  
axis equal; 
title('toroidal helix');


%CONSTANTS
sigma = 0.1;
t = 1;
nDims = 2;


%BUILDING GRAPHS
pairDist = squareform(pdist(x));
K = exp(pairDist.^2./(-2*sigma^2));


%CONSTRUCTING MARKOV CHAIN
d = sum(K,2);
D = diag(d);
P = D\K;
threshold = 1e-5;
P = sparse(P.*double(P>threshold));


%SVD
[~, S, V] = svds(P,nDims+1);
psi=V./(V(:,1)*ones(1,nDims+1));
lambda = sum(S,2);

psi = psi(:, 2:end);
lambda = lambda(2:end);


%DIFFUSION MAPPING 
PSI = psi.*(lambda.^t)';
figure();
subplot(1,2,1)
scatter3(x(:,1), x(:,2), x(:,3), 20, theta, 'filled'); axis equal
title('data')
subplot(1,2,2)
scatter(PSI(:,1), PSI(:,2), 20, theta); axis equal; grid on
title('embedding'); 
xlabel('\lambda_1^t\psi_1(x_i)'); ylabel('\lambda_2^t\psi_2(x_i)')



