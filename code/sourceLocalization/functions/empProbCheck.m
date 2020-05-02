function [out_potential] = empProbCheck(sub_error, align_resid,misalign_resid)

    out_potential = zeros(3,3);
    out_potential(3,:) = ones(1,3).*(1/3);
    numEdges = 10;

    [bin_aligns,al_edges] = discretize(rmoutliers(align_resid),50);
    [bin_misaligns,mis_edges] = discretize(rmoutliers(misalign_resid),50);
    
    al_hist = hist(bin_aligns,14)./sum(hist(bin_aligns));
    al_edges = al_edges(1:5:end);
    mis_hist = hist(bin_misaligns)./sum(hist(bin_misaligns));
    mis_edges = mis_edges(1:5:end);
    for edg = 1:numEdges
        if sub_error < al_edges(edg)
            out_potential(1,1) = al_hist(edg);
            break
        else
            out_potential(1,1) = al_hist(end);
        end
    end
    for edg = 1:numEdges
        if sub_error < mis_edges(edg)
            out_potential(2,2) = mis_hist(edg);
            break
        else
            out_potential(2,2) = mis_hist(end);
        end
    end
    out_potential(1,3) = 1-out_potential(1,1);
    out_potential(2,3) = 1-out_potential(2,2);
end