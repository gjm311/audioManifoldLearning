clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

load('./mat_results/threshTestResults')
thresh_check = thresh_check(:,1);
p_fails = mean(p_fails,2);
p = polyfit(threshes', thresh_check, 2);
f = polyval(p,threshes);


figure(1)
bar(threshes, thresh_check)
hold on 
plot(threshes, f, 'LineWidth',2)
title(sprintf('Movement Detections Flagged For Different Probability Thresholds\n (11 Trials per Threshold Simulated w/Varying Array Shifts)\n[Shifts: 0 - 0.5m by 5cm increments]'))
xlabel('Probability Threshold')
ylabel('Frequency of Flags')
ylim([0 max(thresh_check)+5])


figure(2)
bar(radii,p_fails)
title(sprintf('Average Probability of Failure for Varying Array Shifts\n (trials per shift = 101)'))
xlabel('Array Shift From Location During Training (m)')
ylabel(sprintf('Probability of Failure \n(via MRF Detection)'))



% for r = 1:size(p_fails,1)
%     figure(1)
%     hold on
% %     plot(threshes, local_fails(r,:))
%     histogram(thresh_check)
% 
%     figure(2)
%     plot(threshes, p_fails(r,:))
%     
% end