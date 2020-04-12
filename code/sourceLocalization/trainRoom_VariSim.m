%{
This program estimates the mean and standard devaition of the source
localization algorithm.
%}
 
% % load desired scenario (scroll past commented section) or uncomment and simulate new environment RTFs 
clear
addpath ./RIR-Generator-master
addpath ./functions
addpath ../../../
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

% ---- TRAINING DATA ----
% room setup
disp('Setting up the room');
load('mat_outputs/monoTestSource_biMicCircle_5L300U_2')
load('mat_outputs/biMicCircle_5L300U_monoNode_2')

%---- with optimal params estimate test position ----
num_samples = 1000;
diffs = zeros(1,num_samples);

modelMeans = zeros(1,num_ts);
modelSds = zeros(1,num_ts);
naiMeans = zeros(1,num_ts);
naiSds = zeros(1,num_ts);
subNaiMeans = zeros(1,num_ts);
subNaiSds = zeros(1,num_ts);

for t = 1:num_ts
    
    T60 = T60s(t);
%     RTF_train = reshape(RTF_trains(t,:,:,:),[nD,rtfLen,numArrays]);
    RTF_train = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);  
    scales = scales_t(t,:);
%     gammaL = reshape(gammaLs(t,:,:), [nL,nL]);
    [~,sigmaL] = trCovEst(nL, nD, numArrays, RTF_train, kern_typ, scales_t(t,:));
    gammaL = inv(sigmaL + diag(ones(1,nL).*vari));
    gammaLs(t,:,:) = gammaL;
    micRTF_train = reshape(micRTF_trains(t,:,:,:,:),[numArrays,nD,rtfLen,numMics]);
    micScale = reshape(micScales(t,:,:), [numArrays,numMics]);
    micGammaL = reshape(micGammaLs(t,:,:,:), [numArrays,nL,nL]);
    
    mean_naives = zeros(numArrays,1);
    sub_naives = zeros(numArrays,1);
    naive_p_hat_ts = zeros(numArrays,3);
    
    for n = 1:num_samples
        try
            wavs = dir('./shortSpeech/');
            rand_wav = randi(25);
            file = wavs(rand_wav+2).name;
            [x_tst,fs_in] = audioread(file);
            [numer, denom] = rat(fs/fs_in);
            x_tst = resample(x_tst,numer,denom);
            x_tst = x_tst'; 
        catch
            continue
        end
        
        sourceTest = randSourcePos(1, roomSize, radiusU*.35, ref);
        [~,~, p_hat_t] = test(x_tst, gammaL, RTF_train, micsPos, rirLen, rtfLen, numArrays,...
                    numMics, sourceTrain, sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, scales);
        diffs(n) = mean(mean(((sourceTest-p_hat_t).^2)));
        
        %Naive Estimates
        %Sub-network Estimates
        sub_p_hat_ts = zeros(numArrays, 3); 
        for k = 1:numArrays
            [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train);
            [~,~,sub_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, numArrays-1, numMics, sourceTrain,...
                sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
        end
        
        %Mono node Estimates
        for k = 1:numArrays
            subnet = micsPos((k-1)*numMics+1:k*numMics,:);
            subscales = micScale(k,:);
            gammaL = reshape(micGammaL(k,:,:),[nL,nL]);
            trRTF = reshape(micRTF_train(k,:,:,:), [nD, rtfLen, numMics]);
            [~,~,naive_p_hat_ts(k,:)] = test(x, gammaL, trRTF, subnet, rirLen, rtfLen, 1, numMics, sourceTrain,...
                sourceTest, nL, nU, roomSize, T60, c, fs, kern_typ, subscales);  
            sub_naives(k) = sub_naives(k) + mean((naive_p_hat_ts(k,:)-sub_p_hat_ts(k,:)).^2);
        end
        mean_naives(1) = mean_naives(1) + mean(pdist(naive_p_hat_ts(2:4,:)));
        mean_naives(2) = mean_naives(2)+ mean(pdist([naive_p_hat_ts(1,:); naive_p_hat_ts(3:4,:)]));
        mean_naives(3) = mean_naives(3)+ mean(pdist([naive_p_hat_ts(1:2,:); naive_p_hat_ts(4,:)]));
        mean_naives(4) = mean_naives(4)+ mean(pdist(naive_p_hat_ts(1:3,:)));
    end
    mean_naives = mean_naives./num_samples;
    sub_naives = sub_naives./num_samples;
    
    modelMeans(t) = mean(diffs);
    modelSds(t) = std(diffs);
    naiMeans(t) = mean(mean_naives);
    naiSds(t) = std(mean_naives);
    subNaiMeans(t) = mean(sub_naives);
    subNaiSds(t) = std(sub_naives);
end

save('mat_results/vari_per_t60.mat', 'modelMeans', 'modelSds', 'naiMeans', 'naiSds','subNaiMeans', 'subNaiSds')