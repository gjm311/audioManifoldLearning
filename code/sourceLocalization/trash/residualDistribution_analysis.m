%{
This program displays the distribution of the residual errors 
for the different latent class types in varying room envirnoment; i.e. 
varying noise and T60s. Classes include aligned (implies arrays in same 
position as when ground truth measured) and misaligned (random array moved
to random position in room).
%}


addpath './mat_outputs'

load('resEsts.mat')

aligns = abs(reshape(align_resids,[1,numArrays*size(align_resids,2)]))./sum(abs(reshape(align_resids,[1,numArrays*size(align_resids,2)])));
misaligns = abs(reshape(misalign_resids,[1,numArrays*size(misalign_resids,2)]))./sum(abs(reshape(misalign_resids,[1,numArrays*size(misalign_resids,2)])));
unks = abs(reshape(unk_resids,[1,numArrays*size(unk_resids,2)]))./sum(abs(reshape(unk_resids,[1,numArrays*size(unk_resids,2)])));

% edges = zeros(1,1000);
count = 1;
stop = 1;
skip = .0005;
start = skip;
rng = start:skip:stop;
pdfs = zeros(3,stop/skip);

for r = 1:size(pdfs,2)
    mnRng = rng(r)- skip;
    mxRng = rng(r);
    for j = 1:3
        if j == 1
           curr = aligns;
        end
        if j == 2
           curr = misaligns;
        end
        if j == 3
            curr = unks;
        end
        for i = 1:size(aligns,2)
            if and(curr(i) < mxRng, curr(i) >= mnRng)
                pdfs(j,r) = pdfs(j,r) + 1;
            end
        end
        pdfs(j,r) = pdfs(j,r)/size(aligns,2); 
    end   
end

ticks = 0:.05:1;
align_pdfs = (pdfs(1,pdfs(1,:)>0)); 
misalign_pdfs = (pdfs(2,pdfs(2,:)>0)); 

figure(1)
bar(align_pdfs)
title('Aligned Residual Error Distribution')
xlim([0 20])
ylim([0 max(pdfs(1,:))+.1])
set(gca,'XTick',0:20)
z ={ticks};
set(gca,'XTickLabel',z)
xlabel('Residual Error [\times10^{3}]');
ylabel('Probability Density');
ax = gca;
ax.TitleFontSizeMultiplier  = 2;


figure(2)
bar(misalign_pdfs)
title({'Misaligned Residual Error Distribution'})
xlim([0 20])
ylim([0 max(pdfs(2,:))+.1])
set(gca,'XTick',0:20)
set(gca,'XTickLabel',z)
xlabel('Residual Error');
ylabel('Probability Density');
ax = gca;
ax.TitleFontSizeMultiplier  = 2;
% 
% figure(3)
% bar(pdfs(3,:))
% title('Unknown Class [Uniform Distribution]')
% xlim([0 20])
% ylim([0 max(pdfs(3,:))+.1])
% xlabel('Residual Error');
% ylabel('Probability Density');

% figure(1)
% scatter(sortedAligns(:,1), sortedAligns(:,2))
% ylim([0 1])
% figure(2)
% scatter(sortedMisaligns(:,1), sortedMisaligns(:,2))
% ylim([0 1])
% figure(3)
% scatter(sortedUnknowns(:,1), sortedUnknowns(:,2))
% ylim([0 1])
