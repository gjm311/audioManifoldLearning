clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

%{
This program looks at the localization error of networks of nodes compared to the 
shifts in distance of a random array from where they were during training of the 
SSGP localizer. 

We also look at the posterior probabilities output from the MRF model, and
difference in estimates between different sub-network configurations 
(i.e. LONO or mono-node comparisons).
%}

%%

%---- Visualize full network estimate error mean and std. for increasing shifts (from moveDetect_pFail_sim.m) ----
load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('./mat_results/localErrorFull_res')
radii = [0 radii];
num_ts=size(T60s,2);
% localError = reshape(mean(mean(localErrors,3),2),[1,16]);
for t = 1:num_ts
    T60 = T60s(t);
    modMean = modelMeans(t);
    modMeans = ones(1,size(localErrors,1)+1)*modMean;
    
    localError = reshape(mean(localErrors(:,t,:),3),[1,16]);

    figure(1)
    grid on
    if t == 1
        noMove1 = plot([radii], [modMeans], '--r');
        hold on
        eBar1 = plot([radii], [modMean localError], 'r');
    elseif t == 2
        noMove2 = plot([radii], [modMeans],'--b');
        eBar2 = plot([radii], [modMean localError],'b');
    else
        noMove3 = plot([radii], [modMeans],'--g');
        eBar3 = plot([radii], [modMean localError],'g');
        legend([eBar1,eBar2,eBar3,noMove1,noMove2,noMove3], 'Dynamic (T60 = 0.2s)','Dynamic (T60 = 0.4s)', 'Dynamic (T60 = 0.6s)', 'Static (T60 = 0.2s)','Static (T60 = 0.4s)', 'Static (T60 = 0.6s)')
    end
%     title(sprintf('Average Error in Source Estimation After Movement\n Comparison with Average Error with no Movement\n[Shifts: 0.05m - 3.05m by 20cm increments]'))
%     title(sprintf('Avg. Localization Error With/Without Movement'))
    xlabel(sprintf('Shift Size (m)'))
    ylabel(sprintf('Estimation Error (m) After Shift'))
    % mz = {[0 round(radii(1:2:end),2)]};
    % xticklabels(mz)
    % set(gca,'XTickLabel',mz)
    ylim([0 0.5])
    xlim([0 3])
end
set(gcf,'color','w')


%%

%---- Visualize posterior probability mean and std. for increasing shifts (from moveDetect_pFail_sim.m) ----

load('mat_outputs/monoTestSource_biMicCircle_5L300U_4')
load('./mat_results/pFail_res_4', 'p_fail', 'p_fails', 'radii')
num_ts = size(p_fails,2);
num_samples = 100;

load('./mat_results/pFail_res_5_post', 'p_fails', 'radii')
localErrors = p_fails;
for t = 1:num_ts
    T60 = T60s(t);    
    localError = reshape(mean(localErrors(:,t,:),3),[1,16]);
    
    figure(2)
    grid on
    if t == 1
        hold on
        eBar1 = plot([radii], [localError], 'r');
    elseif t == 2
        eBar2 = plot([radii], [localError],'b');
    else
        eBar3 = plot([radii], [localError],'g');
       legend([eBar1,eBar2,eBar3], 'Avg. Prob. Movement (T60 = 0.2s)','Avg. Prob. Movement (T60 = 0.4s)', 'Avg. Prob. Movement (T60 = 0.6s)')
    end
%     title(sprintf('Average Probability of Movement for Varying Array Shifts'))
    xlabel(sprintf('Shift Size (m)'))
    ylabel(sprintf('Probability of Movement\n(via MRF Detection)'))
   
    ylim([0 1])
    xlim([0 3])
end
set(gcf,'color','w')

for t = 1:num_ts
    T60 = T60s(t);
    modStd = modelSds(t);
    modStds = ones(1,size(localErrors,1))*modStd;
    localErrorStd = std(reshape(localErrors(:,t,:),[100,16]));

    figure(3)
    grid on
    if t == 1
        hold on
        eBar1 = plot([radii], [localErrorStd], 'r');
    elseif t == 2
        eBar2 = plot([radii], [localErrorStd],'b');
    else
        eBar3 = plot([radii], [localErrorStd],'g');
        legend([eBar1,eBar2,eBar3], 'Std. Prob. Failure (T60 = 0.2s)','Std. Prob. Failure (T60 = 0.4s)', 'Std. Prob. Failure (T60 = 0.6s)')
    end
    title(sprintf('Std. of Probability of Failure for Varying Array Shifts'))
    xlabel(sprintf('Array Shift (m) from Location Before Movement'))
    ylabel(sprintf('Probability of Failure\n(via MRF Detection)'))
    ylim([0 1])
    xlim([0 3])
end

set(gcf,'color','w')
%%
%---- Visualize LONO subnet error mean and std. for increasing shifts (from moveDetect_localErrorNai_sim.m) ----
load('./mat_results/localNaiError_res', 'subLocalErrors', 'radii')
localErrors = subLocalErrors;
for t = 1:num_ts
    T60 = T60s(t);
    
    localError = reshape(mean(localErrors(:,t,:),3),[1,16]);

    figure(4)
    grid on
    if t == 1
        eBar1 = plot([radii], [localError], 'r');
        hold on
    elseif t == 2
        eBar2 = plot([radii], [localError],'b');
    else
        eBar3 = plot([radii], [localError],'g');
        legend([eBar1,eBar2,eBar3], 'T60 = 0.2s','T60 = 0.4s', 'T60 = 0.6s');
    end
    title(sprintf('Average Difference in Source Estimation via LONO Sub-networks'))
    xlabel(sprintf('Array Shift (m) from Location Before Movement'))
    ylabel(sprintf('LONO Sub-network v. Left Out Node\n Estimation Differences (m)'))

    ylim([0 28])
    xlim([0 3])
end
set(gcf,'color','w')

%%

%---- Visualize single node error mean and std. for increasing shifts (from moveDetect_localErrorNai_sim.m) ----
load('./mat_results/localNaiError_res', 'monoLocalErrors', 'radii')
localErrors = monoLocalErrors;
for t = 1:num_ts
    T60 = T60s(t);
    
    localError = reshape(mean(localErrors(:,t,:),3),[1,16]);

    figure(5)
    grid on
    if t == 1
        eBar1 = plot([radii], [localError], 'r');
        hold on
    elseif t == 2
        eBar2 = plot([radii], [localError],'b');
    else
        eBar3 = plot([radii], [localError],'g');
        legend([eBar1,eBar2,eBar3], 'T60 = 0.2s','T60 = 0.4s', 'T60 = 0.6s');
    end
   title(sprintf('Average Difference in Source Estimation via Single Nodes'))
 
    xlabel(sprintf('Array Shift (m) from Location Before Movement'))
    ylabel(sprintf('Single Node Estimation Differences (m)'))

    ylim([.3 3.4])
    xlim([0 3])
end
set(gcf,'color','w')