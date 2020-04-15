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

% load('mat_outputs/biMicCircle_5L300U_monoNode')
% micRTF_train = RTF_train;
% micScales = scales;
% micVaris = varis;
%---- load training data (check mat_trainParams for options)----
load('mat_outputs/monoTestSource_biMicCircle_5L300U_2')
% load('mat_results/vari_t60_data.mat')
% load('mat_outputs/movementOptParams')

% modelSd = .1;

%simulate different noise levels
radii = 0:.5:2; 
its_per = 1;
mic_ref = [3 5.75 1; 5.75 3 1; 3 .25 1; .25 3 1];
accs = struct([]);
wav_folder = dir('./shortSpeech/');
wav_folder = wav_folder(3:27);
inits = 1;
rotation = 0;

transMat = [.65 0.3 0.05; .2 .75 0.05; 1/3 1/3 1/3];
init_var = .2;
lambda = .3;
eMax = .3;
thresh = .3;

t = 1;
ts = [1 4 8];
T60s = [.15 .3 .5];
t_curr = ts(t);
T60 = T60s(t);
RTF_train = reshape(RTF_trains(t_curr,:,:,:), [nD, rtfLen, numArrays]);    
scales = scales_t(t_curr,:);
gammaL = reshape(gammaLs(t_curr,:,:), [nL, nL]);

for riter = 1:size(radii,2)
    acc_curr = struct([]);
    radius_mic = radii(riter);
    if riter == 1
        iters_per = its_per+1;
    else 
        iters_per = its_per;
    end
    for it = 1:iters_per 
        %randomize tst source, new position of random microphone (new
        %position is on circle based off radius_mic), random rotation to
        %microphones, and random sound file (max 4 seconds).
        sourceTest = randSourcePos(1, roomSize, radiusU, ref);
        if inits > 1
            movingArray = randi(numArrays);
            [rotation, micsPosNew] = micRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics);
%             if radius_mic == 0
%                 micsPosNew = micsPos;
%             elseif movingArray == 1
%                 micsPosNew = [micsPos(1:2,:)+[0 radius_mic 0]; micsPos(1+numMics:end,:)];
%             elseif movingArray == numArrays
%                 micsPosNew = [micsPos(1:end-numMics,:); micsPos(7:8,:)+[radius_mic 0 0]];
%             elseif movingArray == 2
%                 micsPosNew = [micsPos(1:numMics,:); micsPos(3:4,:)+[radius_mic 0 0]; micsPos(2*numMics+1:end,:)];
%             else
%                 micsPosNew = [micsPos(1:numMics*2,:); micsPos(5:6,:)+[0 radius_mic 0]; micsPos(3*numMics+1:end,:)];
%             end
                % randMicPos = mic_ref(1,:)+[0 radius_mic 0];
    %             new_x1 = randMicPos(:,1)+.025;
    %             new_y1 = randMicPos(:,2);
    %             new_x2 = randMicPos(:,1)-.025;
    %             new_y2 = randMicPos(:,2);
    %             micsPosNew = [new_x1 new_y1 1;new_x2 new_y2 1; micsPos(1+numMics:end,:)];

        else
            movingArray = randi(numArrays);
            micsPosNew = micsPos;
            inits = 2;
        end
        
        rand_wav = randi(numel(wav_folder));
        file = wav_folder(rand_wav).name;
        [x_tst,fs_in] = audioread(file);
        [numer, denom] = rat(fs/fs_in);
        x_tst = resample(x_tst,numer,denom);
        x_tst = x_tst';
       
        %---- Initialize subnet estimates of training positions ----
        sub_p_hat_ts = zeros(numArrays, 3); 
        for k = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
            [~,~,sub_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
        end
        [self_sub_p_hat_ts, p_fail, posteriors] = moveDetectorOpt(x_tst, transMat, init_var, lambda, eMax, thresh, gammaL, numMics, numArrays, micsPosNew, 1, 0, sub_p_hat_ts, scales, RTF_train,...
                rirLen, rtfLen, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ);
            
    %---- estimate test positions before movement ----
        [~,~, pre_p_hat_t] = test(x_tst, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                        numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);

%     %---- estimate test positions after movement ----
        [~,~, p_hat_t] = test(x_tst, gammaL, RTF_train, micsPosNew, rirLen, rtfLen, numArrays,...
                        numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
        
%         mean_naives = zeros(numArrays,1);
%         sub_naives = zeros(numArrays,1);
%         naive_p_hat_ts = zeros(numArrays,3);
%         for k = 1:numArrays
% %                 [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
%             subnet = micsPos((k-1)*numMics+1:k*numMics,:);
%             subscales = micScales(k,:);
%             gammaL = reshape(gammaLs(k,:,:),[nL,nL]);
%             trRTF = reshape(micRTF_train(k,:,:,:), [nD, rtfLen, numMics]);
%             [~,~,naive_p_hat_ts(k,:)] = test(x_tst, gammaL, trRTF, subnet, rirLen, rtfLen, 1, numMics, sourceTrain,...
%                 sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
%             sub_naives(k) = mean((naive_p_hat_ts(k,:)-sub_p_hat_ts(k,:)).^2);
%         end
%         
%         mean_naives(1) = mean(pdist(naive_p_hat_ts(2:4,:)));
%         mean_naives(2) = mean(pdist([naive_p_hat_ts(1,:); naive_p_hat_ts(3:4,:)]));
%         mean_naives(3) = mean(pdist([naive_p_hat_ts(1:2,:); naive_p_hat_ts(4,:)]));
%         mean_naives(4) = mean(pdist(naive_p_hat_ts(1:3,:)));
%         
%         %Here we compare single node estimates to that of the sub-net
%         %without the single node.
%         sub_naives
%         mean_naive
%         naive_p_hat_ts
                   
        %naive estimate calculated based off variance of new
        %positional estimates and ones taken in training. If estimate sd greater than
        %standard deviation of error of static model then we predict
        %movement.
        naive_p_fail = norm(p_hat_t - pre_p_hat_t);
        naive2_p_fail = mean(std(self_sub_p_hat_ts- sub_p_hat_ts));
        %PLOT
        f = plotRoom(roomSize, micsPosNew, sourceTrain, sourceTest, nL, p_hat_t);
        title(sprintf('Detecting Array Movement \n [Array Movement: %s m]', num2str(radius_mic)))
        naiveTxt = text(.6,.5, sprintf('Network Estimate Distance:\n %s',num2str(round(naive_p_fail,2))));
        naive2Txt = text(2.6,.5, sprintf('Std of Sub-Network Estimate Diff.:\n %s',num2str(round(naive2_p_fail,2))));
        binTxt = text(4.6,.5, sprintf('MRF Based Probability:\n %s',num2str(round(p_fail,2))));
        %     %Uncomment below for pauses between iterations only continued by
    %     %clicking graph  
        w = waitforbuttonpress;
        delete(binTxt);
        delete(naiveTxt);
        delete(naive2Txt);
        clf
         
    end
   
end
