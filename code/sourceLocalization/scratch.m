save('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
save('mat_outputs/vari_per_t60_4', 'modelSds', 'modelMeans', 'T60s', 'num_samples')

T60s = [.2 .4 .6];
ts = [2 6 10];

RTF_train_curr = zeros(3, nD, rtfLen, numArrays);
scales_curr = zeros(3, numArrays);
gammaL_curr = zeros(3, nL, nL);
modelMeanCurr = zeros(1,3);
modelSdCurr = zeros(1,3);

for t = 1:size(ts,2)
    t_curr = ts(t);
    RTF_train_curr(t,:,:,:) = reshape(RTF_trains(t_curr,:,:,:), [nD, rtfLen, numArrays]);    
    scales_curr(t,:) = scales_t(t_curr,:);
    gammaL_curr(t,:,:) = reshape(gammaLs(t_curr,:,:), [nL, nL]);
    modelMeanCurr(t) = modelMeans(t_curr);
    modelSdCurr(t) = modelSds(t_curr);
end

RTF_trains = RTF_train_curr;
scales_t = scales_curr;
gammaLs = gammaL_curr;
modelMeans = modelMeanCurr;
modelSds = modelSdCurr;

clear RTF_train_curr
clear scales_curr
clear gammaL_curr
clear modelMeanCurr
clear modelSdCurr