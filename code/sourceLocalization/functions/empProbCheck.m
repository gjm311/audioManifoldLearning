function [out_probs] = empProbCheck(sub_error, align_resids,misalign_resids)
    out_probs = zeros(3,3);
    out_probs(3,:) = ones(1,3).*(1/3);
    numEdges = 10;

    [bin_aligns,al_edges] = discretize(rmoutliers(align_resids(r,:)),50);
    [bin_misaligns,mis_edges] = discretize(rmoutliers(misalign_resids(r,:),'mean'),50);
    
    al_hist = hist(bin_aligns)./sum(hist(bin_aligns));
    al_edges = al_edges(1:5:end);
    mis_hist = bar(hist(bin_misaligns)./sum(hist(bin_misaligns)));
    mis_edges = mis_edges(1:5:end);
    for edg = 1:numEdges
        if sub_error < al_edges(edg)
            out_probs(1,1) = al_hist(edg);
            break
        else
            out_probs(1,1) = al_hist(end);
        end
    end
    for edg = 1:numEdges
        if sub_error < mis_edges(edg)
            out_probs(2,2) = mis_hist(edg);
            break
        else
            out_probs(2,2) = mis_hist(end);
        end
    end
    out_probs(1,3) = 1-out_probs(1,1);
    out_probs(2,3) = 1-out_probs(2,2);
end