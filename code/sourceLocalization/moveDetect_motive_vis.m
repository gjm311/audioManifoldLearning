clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
This program looks at the localization error (MSE) and posterior probabilities
output from the MRF model when arrays are moved varying distances from where 
they were during training). Results also show the benefit of using the proabilities 
of failure of the MRF based movement detector rather than localization
error.
%}


%---- Uncomment for visualizing posterior probability results ----
load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('./mat_results/pFail_res_5', 'p_fail', 'p_fails', 'radii')
num_ts = size(p_fails,2);
num_samples = 100;

pfailSds = std(reshape(p_fails(:,t,:),[100,16]));
sem = pfailSds./sqrt(num_samples);
pfail_ci95 = sem*tinv(.975, num_samples-1);

figure(1)
errorbar([0 radii], [0; p_fail], [0 pfail_ci95])
% plot([0;p_fail])
title(sprintf('Average Probability of Failure for Varying Array Shifts\n Confidence Intervals: 95%%\n[Shifts: 0.05m - 3.05m by 20cm increments]'))
xlabel(sprintf('Array Shift (m) from Location During Training'))
ylabel(sprintf('Probability of Failure \n (via MRF Detection)'))
% mz = {[round(radii(1:2:end),2)]};
% xticklabels([0 radii])
% set(gca,'XTickLabel',mz)
ylim([0 1])


% ---- Uncomment for localization results for increasing shifts.
load('./mat_results/localErrorFull_res', 'localErrors', 'radii')
localErrors = p_fails;
% localError = reshape(mean(mean(localErrors,3),2),[1,16]);
for t = 1:num_ts
    T60 = T60s(t);
    modMean = modelMeans(t);
    modMeans = ones(1,size(localErrors,1))*modMean;
    
    localError = reshape(mean(localErrors(:,t,:),3),[1,16]);
    
%     semLE = localErrorSds./sqrt(num_samples);
%     localError_ci95 = semLE*tinv(.975, num_samples-1);

    figure(2)
    if t == 1
        noMove1 = plot([radii], [modMeans], '--r');
        hold on
        eBar1 = plot([radii], [localError], 'r');
    elseif t == 2
        noMove2 = plot([radii], [modMeans],'--b');
        eBar2 = plot([radii], [localError],'b');
    else
        noMove3 = plot([radii], [modMeans],'--g');
        eBar3 = plot([radii], [localError],'g');
        legend([eBar1,eBar2,eBar3,noMove1,noMove2,noMove3], 'Avg. Error (T60 = 0.2s)','Avg. Error (T60 = 0.4s)', 'Avg. Error (T60 = 0.6s)', 'Avg. Error No Movement (T60 = 0.2s)','Avg. Error No Movement (T60 = 0.4s)', 'Avg. Error No Movement (T60 = 0.6s)')
    end
    title(sprintf('Average Error in Source Estimation After Movement\n Comparison with Average Error with no Movement\n[Shifts: 0.05m - 3.05m by 20cm increments]'))
    xlabel(sprintf('Array Shift (m) from Location Before Movement'))
    ylabel(sprintf('Estimation Error (m) After Shift'))
    % mz = {[0 round(radii(1:2:end),2)]};
    % xticklabels(mz)
    % set(gca,'XTickLabel',mz)
    ylim([0 1])
    xlim([0 3])
end

for t = 1:num_ts
    T60 = T60s(t);
    modStd = modelSds(t);
    modStds = ones(1,size(localErrors,1))*modStd;
    localErrorStd =std(reshape(localErrors(:,t,:),[100,16]));
%     semLE = localErrorSds./sqrt(num_samples);
%     localError_ci95 = semLE*tinv(.975, num_samples-1);

    figure(3)
    if t == 1
        noMove1 = plot([radii], [modStds], '--r');
        hold on
        eBar1 = plot([radii], [localErrorStd], 'r');
    elseif t == 2
        noMove2 = plot([radii], [modStds],'--b');
        eBar2 = plot([radii], [localErrorStd],'b');
    else
        noMove3 = plot([radii], [modStds],'--g');
        eBar3 = plot([radii], [localErrorStd],'g');
        legend([eBar1,eBar2,eBar3,noMove1,noMove2,noMove3], 'Avg. Error (T60 = 0.2s)','Avg. Error (T60 = 0.4s)', 'Avg. Error (T60 = 0.6s)', 'Avg. Error No Movement (T60 = 0.2s)','Avg. Error No Movement (T60 = 0.4s)', 'Avg. Error No Movement (T60 = 0.6s)')
    end
    title(sprintf('Std. of Error in Source Estimation After Movement\n Comparison with Std. of Error with no Movement\n[Shifts: 0.05m - 3.05m by 20cm increments]'))
    xlabel(sprintf('Array Shift (m) from Location Before Movement'))
    ylabel(sprintf('Estimation Error (m) After Shift'))
    % mz = {[0 round(radii(1:2:end),2)]};
    % xticklabels(mz)
    % set(gca,'XTickLabel',mz)
    ylim([0 1])
    xlim([0 3])
end


% ---- Uncomment for localization results for increasing shifts.
load('./mat_results/localNaiError_res', 'monoLocalErrors', 'radii')
localErrors = monoLocalErrors;
% localError = reshape(mean(mean(localErrors,3),2),[1,16]);
for t = 1:num_ts
    T60 = T60s(t);
    
    localError = reshape(mean(localErrors(:,t,:),3),[1,16]);
%     localErrorSds = std(reshape(localErrors(:,t,:),[100,16]));
%     semLE = localErrorSds./sqrt(num_samples);
%     localError_ci95 = semLE*tinv(.975, num_samples-1);

    figure(4)
    if t == 1
        eBar1 = plot([radii], [localError], 'r');
        hold on
    elseif t == 2
        eBar2 = plot([radii], [localError],'b');
    else
        eBar3 = plot([radii], [localError],'g');
        legend([eBar1,eBar2,eBar3], 'Avg. Diff. (T60 = 0.2s)','Avg. Diff. (T60 = 0.4s)', 'Avg. Diff. (T60 = 0.6s)');
    end
    title(sprintf('Average Difference in Source Estimation of Single Nodes\n One Node Shifted per Iteration\n[Shifts: 0.05m - 3.05m by 20cm increments]'))
    xlabel(sprintf('Array Shift (m) from Location Before Movement'))
    ylabel(sprintf('Single Node Estimation Differences (m)'))
    % mz = {[0 round(radii(1:2:end),2)]};
    % xticklabels(mz)
    % set(gca,'XTickLabel',mz)
    ylim([.3 3.4])
    xlim([0 3])
end


% ---- Uncomment for localization results for increasing shifts.
load('./mat_results/localNaiError_res', 'subLocalErrors', 'radii')
localErrors = subLocalErrors;
% localError = reshape(mean(mean(localErrors,3),2),[1,16]);
for t = 1:num_ts
    T60 = T60s(t);
    
    localError = reshape(mean(localErrors(:,t,:),3),[1,16]);
%     localErrorSds = std(reshape(localErrors(:,t,:),[100,16]));
%     semLE = localErrorSds./sqrt(num_samples);
%     localError_ci95 = semLE*tinv(.975, num_samples-1);

    figure(5)
    if t == 1
        eBar1 = plot([radii], [localError], 'r');
        hold on
    elseif t == 2
        eBar2 = plot([radii], [localError],'b');
    else
        eBar3 = plot([radii], [localError],'g');
        legend([eBar1,eBar2,eBar3], 'Avg. Diff. (T60 = 0.2s)','Avg. Diff. (T60 = 0.4s)', 'Avg. Diff. (T60 = 0.6s)');
    end
    title(sprintf('Average Difference in Source Estimation via LONO Sub-networks \nand Left Out Nodes (One Node Shifted per Iteration)\n[Shifts: 0.05m - 3.05m by 20cm increments]'))
    xlabel(sprintf('Array Shift (m) from Location Before Movement'))
    ylabel(sprintf('LONO Sub-network v. Left Out Node\n Estimation Differences (m)'))
    % mz = {[0 round(radii(1:2:end),2)]};
    % xticklabels(mz)
    % set(gca,'XTickLabel',mz)
    ylim([0 28])
    xlim([0 3])
end