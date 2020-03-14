clear
addpath ./RIR-Generator-master
addpath ./functions
mex -setup c++
mex RIR-Generator-master/rir_generator.cpp;
addpath ./stft
addpath ./shortSpeech

load('./mat_results/threshTestResults')


for r = 1:size(p_fails,1)
    figure(1)
    hold on
    plot(threshes, local_fails(r,:))

    figure(2)
    plot(threshes, p_fails(r,:))
    
end