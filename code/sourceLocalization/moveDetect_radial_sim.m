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
% ---- Initialize Parameters for training ----

%---- load training data (check mat_trainParams for options)----
load('mat_outputs/monoTestSource_biMicCircle_5L50U.mat')
load('mat_outputs/movementOptParams')

%---- Initialize subnet estimates of training positions ----
randL = randi(numArrays, 1);
holderTest = sourceTrain(randL,:);
for k = 1:numArrays
    [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
    [~,~,sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
        holderTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
end

%simulate different noise levels
radii = [0:.005:.03];
iters_per = 10;
mic_ref = [3 5 1; 5 3 1; 3 1 1; 1 3 1];
accs = struct([]);
   
for riter = 1:size(radii)
    
    sourceTest = randSourcePos(1, roomSize, radiusU, ref);
    source = rand;
    radius_mic = radii(riter);
    movingArray = randi(numArrays,1);
    %******
    randMicsPos = randMicsPos(1, roomSize, radius_mic, mics_ref(movingArray,:));
    rotation = rand*360;
    new_x1 = randMicsPos(:,1)+.025*cos(rotation);
    new_y1 = randMicsPos(:,2)+.025*sin(rotation);
    new_x2 = randMicsPos(:,1)-.025*cos(rotation);
    new_y2 = randMicsPos(:,2)-.025*sin(rotation);
    if movingArray == 1
        micsPosNew = [new_x1 new_y1 1;new_x2 new_y2 1; micsPos(1+numMics:end,:)];
    elseif movingArray == numArrays
        micsPosNew = [micsPos(1:end-numMics,:); new_x1 new_y1 1; new_x2 new_y2 1];
    elseif movingArray == 2
        micsPosNew = [micsPos(1:numMics,:); new_x1 new_y1 1; new_x2 new_y2 1; micsPos(2*numMics+1:end,:)];
    else
        micsPosNew = [micsPos(1:numMics*2,:); new_x1 new_y1 1; new_x2 new_y2 1; micsPos(3*numMics+1:end,:)];
    end
    
    for it = 1:iters_per
        %initialize ground truths
        
        acc_curr = struct([]);
        t_count = 0;

        for t = 1:numMovePoints
            if mod(t,samplerate) == 0
                t_count = t_count + 1;
               

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
                    [self_sub_p_hat_ts, p_fail, posteriors] = moveDetectorOpt(x, opt_trans_mat, opt_vars, opt_lambdas, opt_eMaxs, opt_threshs, gammaL, numMics, numArrays, micsPosNew, t_count, p_fail, sub_p_hat_ts, scales, RTF_train,...
                    rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);
                else
                    [self_sub_p_hat_ts, p_fail, posteriors] = moveDetectorOpt(x, opt_trans_mat, opt_vars, opt_lambdas, opt_eMaxs, opt_threshs, gammaL, numMics, numArrays, micsPosNew, t_count, p_fail, self_sub_p_hat_ts, scales, RTF_train,...
                    rirLen, rtfLen, sourceTrain, holderTest, nL, nU, roomSize, T60, c, fs, kern_typ);
                end

%                 acc_curr(t_count).latent_acc = norm(gt_latents(:,:,t_count)-posteriors);
                latents = round(posteriors);

                %IF movement detected, estimate test position using subnet (w/o moving array).
                %Update parameters and covariance via Bayes with params derived from subnetwork.
                if p_fail > opt_thresh
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
end


%simulate minimal movement

%simulate multiple arrays moving

%min/max number of labelled estimates needed