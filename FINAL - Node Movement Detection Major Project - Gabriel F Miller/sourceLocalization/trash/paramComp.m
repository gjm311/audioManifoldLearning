
%---- load training data (check mat_trainParams for options)----
load('mat_trainParams/biMicCircle_5L50U.mat')
addpath ./functions

%Initialize hyper-parameters (namely, gaussian kernel scalar
%value and noise variance estimate)
kern_typ = 'gaussian';
tol = 10e-3;
alpha = 10e-3;
max_iters = 100;
init_scales = 1*ones(1,numArrays);
var_init = 0;
sigmaL = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, init_scales);


cov = zeros(nD,nD);
for r = 1:nD
    for l = 1:nD
        kern = 0;
        for i = 1:nD        
            for q = 1:numArrays
                for w = 1:numArrays
                    kernQ = kernel(RTF_train(r,:,q), RTF_train(i,:,q), kernTyp, scales(q));
                    kernW = kernel(RTF_train(l,:,w), RTF_train(i,:,w), kernTyp, scales(w));
                    kern = kern + kernQ*kernW;            
                end
            end
        end
        cov(r,l) = (1/numArrays^2)*kern;
    end
end