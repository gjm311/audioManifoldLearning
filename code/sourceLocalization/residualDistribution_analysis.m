%{
This program displays the distribution of the residual errors 
for the different latent class types in varying room envirnoment; i.e. 
varying noise and T60s. Classes include aligned (implies arrays in same 
position as when ground truth measured) and misaligned (random array moved
to random position in room).
%}

addpath './mat_outputs'
load('resEsts.mat')

[bin_aligns,al_edges] = discretize(rmoutliers(align_resids),50);
[bin_misaligns,mis_edges] = discretize(rmoutliers(misalign_resids,'mean'),50);


figure(1)
h = bar(hist(bin_aligns)./sum(hist(bin_aligns)));
title('Aligned Distribution')
az ={round(al_edges,3)};
set(gca,'XTickLabel',az)
ylim([0,1])
xlabel('Min. Bin Error');
ylabel('Probability Density');
ax = gca;
ax.TitleFontSizeMultiplier  = 2;

figure(2)
h = bar(hist(bin_misaligns)./sum(hist(bin_misaligns)));
title({'Misaligned Distribution'})
mz = {round(mis_edges,3)};
set(gca,'XTickLabel',mz)
ylim([0,1])
xlabel('Min. Bin Error');
ylabel('Probability Density');
ax = gca;
ax.TitleFontSizeMultiplier  = 2;

% % edges = zeros(1,1000);
% count = 1;
% stop = 1;
% skip = .01;
% start = skip;
% rng = start:skip:stop;
% pdfs = zeros(2,stop/skip);
% 
% for r = 1:size(pdfs,2)
%     mnRng = rng(r)- skip;
%     mxRng = rng(r);
%     for j = 1:2
%         if j == 1
%            curr = aligns;
%            for i = 1:size(aligns,2)
%                 if and(curr(i) < mxRng, curr(i) >= mnRng)
%                     pdfs(j,r) = pdfs(j,r) + 1;
%                 end
%             end
%   
%         end
%         if j == 2
%            curr = misaligns;
%            for i = 1:size(curr,2)
%                 if and(curr(i) < mxRng, curr(i) >= mnRng)
%                     pdfs(j,r) = pdfs(j,r) + 1;
%                 end
%             end
%         end
%     end   
% end

% % aq = 1.5:.45:10.5;
% align_pdfs = pdfs(1,pdfs(1,:)>0); 
% % mq = 1.5:.8:14.5;
% misalign_pdfs = pdfs(2,pdfs(2,:)>0); 
% a_ticks = 0:(1/size(align_pdfs,2)):1;
% m_ticks = 0:(1/size(misalign_pdfs,2)):1;
% % align_pdfs = interp1(align_pdfs,aq);
% % misalign_pdfs = interp1(misalign_pdfs,mq);

% figure(1)
% bar(align_pdfs)
% title('Aligned Residual Error Distribution')
% % xlim([0 20])
% ylim([0 max(pdfs(1,:))+.1])
% set(gca,'XTick',0:size(align_pdfs,2))
% z ={round(a_ticks,2)};
% set(gca,'XTickLabel',z)
% xlabel('Residual Error');
% ylabel('Probability Density');
% ax = gca;
% ax.TitleFontSizeMultiplier  = 2;


% figure(2)
% bar(misalign_pdfs)
% title({'Misaligned Residual Error Distribution'})
% % xlim([0 20])
% ylim([0 max(pdfs(2,:))+.1])
% set(gca,'XTick',0:size(misalign_pdfs,2))
% z = {round(m_ticks,2)};
% set(gca,'XTickLabel',z)
% xlabel('Residual Error');
% ylabel('Probability Density');
% ax = gca;
% ax.TitleFontSizeMultiplier  = 2;


