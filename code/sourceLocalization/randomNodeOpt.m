%This program is used to optimize parameters in the MRF approach to
%recognizing misalignment in an array network. Optimization is done via a 
%random search method where by we randomly select parameters (variance, 
%lambda, and psi from the paper "Misalignment Recognition Using MRFs with 
%Fully connected Latent Variables for Detecting Localization Failures" by 
%Naoki Akai, Luis Yoichi Morales et al.

clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
%---- load training data (check mat_trainParams for options)----
load('mat_outputs/monoTestSource_biMicCircle_5L50U.mat')

%---- Initialize storage parameters ----
sourceTest = [3.5, 3.5, 1];
% Q_t = inv(sigmaL+rand*10e-3*eye(size(sigmaL)));
%---- Initialize bayes parameters (via Van Vaerenbergh method) ----
[mu, cov, Q] = bayesInit(nD, sourceTrain, RTF_train, kern_typ, scales, numArrays, vari);
Q_t = Q(1:nL,1:nL);
RTF_test =  rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTest, roomSize, T60, rirLen, c, fs);
sigma_Lt = tstCovEst(nL, nD, numArrays, RTF_train, RTF_test, kern_typ, scales); 

q_t_new = zeros(nL,1);
h_t_new = zeros(nL,1);
p_fail = 0;
posterior = zeros(numArrays, 3);
sub_p_hat_ts = zeros(numArrays, 3); 
self_sub_p_hat_ts = zeros(numArrays, 3);
mvFlag = 0;
numStr = 1; upd = 0;

%---- Initialize moving mic line params ----
movingArray = 1;
init_pauses = 2;
moving_iters = 4;
numMovePoints = moving_iters + init_pauses;
height = 1;
movingMicsPos = [micLine(micsPos(movingArray,:), [.5,5.5,1], moving_iters), height*ones(moving_iters,1)];
movingMicsPos = [[3 5 1].*ones(init_pauses,3);movingMicsPos];

%---- Initialize subnet estimates of training positions ----
randL = randi(numArrays, 1);
holderTest = sourceTrain(randL,:);
for k = 1:numArrays
    [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
    [~,~,~,sub_p_hat_ts(k,:)] = test(x, sigma_Lt, gammaL, trRTF, RTF_test, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
        holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
end

%set ground truth for all iterations of moving array
static_truth = [ones(4,1) zeros(4,1) zeros(4,1)];
mvmnt_truth = [1 0 0; 0 1 0; 0 1 0; 0 1 0];
ground_truth_init = repmat(static_truth, [1 1 init_pauses]);
ground_truth_mv = repmat(mvmnt_truth, [1 1 moving_iters]);
ground_truth = cat(3, ground_truth_init, ground_truth_mv);

%set limits of var_A, psi, and lambda
min_thetas = 0;
max_thetas = 1;
min_psi = .5;
max_psi = 1;

%set number of trials to be simulated per iteration
iters = 0:7;
trials = 2.^iters;
str = struct([]);

for iter = iters+1
    curr_str = struct([]);
    for trial = 1:trials(iter)
        psi = zeros(3,3);
        al_rand1 = rand*(max_psi-min_psi)+min_psi; psi(1,1) = al_rand1;
        al_rand2 = (1-al_rand1)*rand; psi(1,2) = al_rand2;
        psi(1,3) = 1-al_rand1-al_rand2;
        mis_rand1 = rand*(max_psi-min_psi)+min_psi; psi(2,2) = mis_rand1;
        mis_rand2 = (1-mis_rand1)*rand; psi(2,1) = mis_rand2;
        psi(2,3) = 1-mis_rand1-mis_rand2;
        psi(3,:) = ones(1,3)*(1/3);
        varA = (max_thetas-min_thetas)*rand+min_thetas;
        lambda = (max_thetas-min_thetas)*rand+min_thetas;
        eMax = (max_thetas-min_thetas)*rand+min_thetas;
        
        curr_str(trial).psi = psi;
        curr_str(trial).varA = varA;
        curr_str(trial).lambda = lambda;
        curr_str(trial).eMax = eMax;
        accs = zeros(1,numMovePoints);
        posteriors = zeros(numArrays,3,numMovePoints);
        for t = 1:numMovePoints
            
            new_x1 = movingMicsPos(t,1)-.025;
            new_x2 = movingMicsPos(t,1)+.025;
            micsPosNew = [new_x1 movingMicsPos(t,2:3);new_x2 movingMicsPos(t,2:3); micsPos(1+numMics:end,:)];

            %Check if movement occured. If no movement detected, residual error
            %of position estimate from subnets calculated from initial
            %estimates. If movmement previously detected, we compare to
            %estimate from previous iteration.
            if mvFlag == 0
                [self_sub_p_hat_ts, p_fail, posterior] = mvOpt(transMat, varA, lambda, eMax, x, sigma_Lt, gammaL, numMics, numArrays, micsPosNew, t, p_fail, sub_p_hat_ts, scales, RTF_train, RTF_test,...
                rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            else
                [self_sub_p_hat_ts, p_fail, posterior] = mvOpt(transMat, varA, lambda, eMax, x, sigma_Lt, gammaL, numMics, numArrays, micsPosNew, t, p_fail, self_sub_p_hat_ts, scales, RTF_train,RTF_test,...
                rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            end
            if p_fail > thresh
                mvFlag = 1;
            else
                mvFlag = 0;
            end
            accs(t) = mse(ground_truth(:,:,t), posterior);
            posteriors(:,:,t) = posterior;
        end
        curr_str(trial).accs = accs;
        curr_str(trial).mean_accs = mean(accs);
        curr_str(trial).posteriors = posteriors;
    end
    str(iter).iters = curr_str;
end
