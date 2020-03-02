
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
T60 = .15;
iters = 3;
Q_t = inv(sigmaL+rand*10e-3*eye(size(sigmaL)));
%---- Initialize bayes parameters (via Van Vaerenbergh method) ----
% [mu, cov, Q_t] = bayesInit(nL, sourceTrainL, RTF_train, kern_typ, scales, numArrays, vari);
q_t_new = zeros(nL,1);
h_t_new = zeros(nL,1);
budgetL = nL;
budgetStr = ceil(nL*1.5);
p_fail = 0;
samplerate = 3;
posteriors = zeros(numArrays, 3);
sub_p_hat_ts = zeros(numArrays, 3); 
self_sub_p_hat_ts = zeros(numArrays, 3);
mvFlag = 0;
rtf_str = zeros(budgetStr,rtfLen, numArrays);
est_str = zeros(budgetStr,3);
numStr = 1; upd = 0;
iter = 0;
height = size(sourceTrainL,3);
pause_opts = samplerate:samplerate:samplerate*3;
move_opts = samplerate*5:samplerate:10*samplerate;

%initialize/define MRF hyperparameters
netThresh_min = .4;
netThresh_max = .6;
thresh_min = .05;
thresh_max = .8;
params_min = 0;
p_max = 1;
trans_min =.5;
trans_max=  1;
num_iters = 100;
trans_mats = zeros(3,3,num_iters);
vars = zeros(num_iters,1);
eMaxs = zeros(num_iters,1);
lambdas = zeros(num_iters,1);
threshs = zeros(num_iters,1);
net_threshs = zeros(num_iters,1);
accs = struct([]);


%---- Initialize subnet estimates of training positions ----
randL = randi(numArrays, 1);
holderTest = sourceTrain(randL,:);
for k = 1:numArrays
    [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
    [~,~,sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
        holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
end


for itr = 1:num_iters
    %---- Initialize moving mic line params ----
    %random durations chosen for pausing in beginning/end and also number of
    %discrete jumps per iteration. Moving array also random. 
    movingArray = randi(numArrays,1);
    init_pauses = pause_opts(randi([1,size(pause_opts,1)],1));
    moving_iters = move_opts(randi([1,size(move_opts,1)],1));
    end_pauses = pause_opts(randi(size(pause_opts,1),1));
    numMovePoints = moving_iters + init_pauses+end_pauses;
    end_pts = [.5,5.5,1; 5.5 5.5 1; 5.5 .5 1; .5 .5 1];
    movingMicsPos = [micLine(micsPos(movingArray,:), end_pts(movingArray,:), moving_iters), height*ones(moving_iters,1)];
    movingMicsPos = [[3 5 1].*ones(init_pauses,3);movingMicsPos];
    movingMicsPos = [movingMicsPos; ones(end_pauses,3).*end_pts(movingArray,:)];

    %initialize ground truths. Note gt_pfail used to optimize threshold,
    %gt_latents used to optimize 
    mvmnt_truth = [1 0 0; 0 1 0; 0 1 0; 0 1 0];
    static_truth = [ones(4,1) zeros(4,1) zeros(4,1)];
    gt_latents = zeros(size(posteriors,1), size(posteriors,2), numMovePoints/samplerate);
    gt_latents(:,:,1:init_pauses/samplerate) = repmat(static_truth, [1,1,init_pauses/samplerate]);
    gt_latents(:,:,init_pauses/samplerate+1:moving_iters/samplerate+init_pauses/samplerate) = repmat(mvmnt_truth, [1,1,moving_iters/samplerate]);
    gt_latents(:,:,init_pauses/samplerate+moving_iters/samplerate+1:end) = repmat(static_truth, [1,1,end_pauses/samplerate]);
    gt_pfail = [zeros(1,init_pauses/samplerate) ones(1,moving_iters/samplerate) zeros(1,init_pauses/samplerate)];
    
    %init random mrf parameters
    net_threshs(itr) = rand*(netThresh_max-netThresh_min)*netThresh_min;
    vars(itr) = rand;
    eMaxs(itr) = rand;
    lambdas(itr) = rand;
    threshs(itr) = rand*(thresh_max-thresh_min)*thresh_min;
    trans_al1 = rand*(trans_max-trans_min)+trans_min;
    trans_mis1 = (1-trans_al1)*rand;
    trans_unk1 = 1 - trans_al1 - trans_mis1;
    trans_mis2 = rand*(trans_max-trans_min)+trans_min;    
    trans_al2 = (1-trans_mis2)*rand;
    trans_unk2 = 1 - trans_al2 - trans_mis2;
    trans_mats(:,:,itr) = [trans_al1 trans_mis1 trans_unk1; trans_al2 trans_mis2 trans_unk2; 1/3 1/3 1/3];
    
    acc_curr = struct([]);
    t_count = 0;
    
    for t = 1:numMovePoints
        if mod(t,samplerate) == 0
            t_count = t_count + 1;
            %At each iteration, add noise to vary position estimates for more
            %variety when updating. For testing labelled points (to detect
            %movement), we store original precision matrix and only update if we
            %update our subnetwork position estimates; i.e. update when new RTF samples
            %taken

            if movingArray == 1
                new_x1 = movingMicsPos(t,1)-.025;
                new_x2 = movingMicsPos(t,1)+.025;
                micsPosNew = [new_x1 movingMicsPos(t,2:3);new_x2 movingMicsPos(t,2:3); micsPos(1+numMics:end,:)];
            elseif movingArray == numArrays
                new_x1 = movingMicsPos(t,2)-.025;
                new_x2 = movingMicsPos(t,2)+.025;
                micsPosNew = [micsPos(1:end-numMics,:); new_x1 movingMicsPos(t,2:3);new_x2 movingMicsPos(t,2:3)];
            else
                if movingArray == 2
                    new_x1 = movingMicsPos(t,2)-.025;
                    new_x2 = movingMicsPos(t,2)+.025;
                else
                    new_x1 = movingMicsPos(t,1)-.025;
                    new_x2 = movingMicsPos(t,1)+.025;
                end
                newMicsPos = [new_x1 movingMicsPos(t,2:3);new_x2 movingMicsPos(t,2:3)];
                micsPosNew = [micsPos(1:(movingArray*numMics)-numMics,:); newMicsPos; micsPos((movingArray*numMics)+1:end,:)];
            end
                
            %Estimate position for all labelled points for all subnetworks and take
            %average probability of movement for each labelled point.
            %Estimate training position for each subnetwork. Calculated when
            %movement stops and new labelled RTF samples added. 
            if upd == 1
                randL = randi(numArrays,1);
                holderTest = sourceTrain(randL,:);
                gammaL = Q_t;
                for k = 1:numArrays
                    [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPosNew, RTF_train);
                    [~,~,sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                        holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
                end
            end

            %Check if movement occured. If no movement detected, residual error
            %of position estimate from subnets calculated from initial
            %estimates. If movmement previously detected, we compare to
            %estimate from previous iteration.
            if mvFlag == 0
                [self_sub_p_hat_ts, p_fail, posteriors] = moveDetectorOpt(x,trans_mats(:,:,itr), vars(itr), lambdas(itr), eMaxs(itr), threshs(itr), gammaL, numMics, numArrays, micsPosNew, t_count, p_fail, sub_p_hat_ts, scales, RTF_train,...
                rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            else
                [self_sub_p_hat_ts, p_fail, posteriors] = moveDetectorOpt(x, trans_mats(:,:,itr), vars(itr), lambdas(itr), eMaxs(itr), threshs(itr), gammaL, numMics, numArrays, micsPosNew, t_count, p_fail, self_sub_p_hat_ts, scales, RTF_train,...
                rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            end
            
            acc_curr(t_count).latent_acc = norm(gt_latents(:,:,t_count)-posteriors);
            latents = round(posteriors);
            
            %IF movement detected, estimate test position using subnet (w/o moving array).
            %Update parameters and covariance via Bayes with params derived from subnetwork.
            if p_fail > net_threshs(itr)
                if gt_pfail(t_count) == 1
                    acc_curr(t_count).pfail_acc = 1;
                else
                    acc_curr(t_count).pfail_acc = 0;                    
                end

                if mvFlag == 0
                   mvFlag = 1;
                end
                upd = 0;
                [sub_scales, num_static, RTF_test, p_hat_t, k_t_new] = subEst(Q_t, posteriors, numMics, numArrays, micsPosNew, RTF_train, scales, x, rirLen, rtfLen,...
                    sourceTrain, sourceTest, nL, nU, T60, c, fs, kern_typ, roomSize);

                %KEY: we store positional estimates taken from sub-network of
                %arrays and test RTF sample measured by ALL arrays (moving
                %included). This way, during update phase, we are updating kernel
                %based off accurate estimates of test position and test RTF sample
                %describing new array network dynamic.
                if latents ~= ones(numArrays, 1)
                    if numStr > budgetStr
                        rtf_str(1,:,:) = RTF_test;
                        rtf_str = circshift(rtf_str, -1, 1);
                        est_str(1,:) = p_hat_t;
                        est_str = circshift(est_str, -1, 1);
                    else
                        rtf_str(numStr,:,:) = RTF_test;
                        est_str(numStr,:) = p_hat_t;
                        numStr = numStr + 1;
                    end
                end               
            else
                if gt_pfail(t_count) == 0
                    acc_curr(t_count).pfail_acc = 1;
                else
                    acc_curr(t_count).pfail_acc = 0;                    
                end
                upd = 0;
                [RTF_test, k_t_new, p_hat_t] = test(x, Q_t, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                        numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
            end
            accs(itr).sims = acc_curr;    
        end
        
    end   
end

save('mat_outputs/movementOptParams')