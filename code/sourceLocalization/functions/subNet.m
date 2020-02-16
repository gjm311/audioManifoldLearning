function [subnet, subscales, trRTF] = subNet(k, numArrays, numMics, scales, micsPos, RTF_train)
    if k == 1
        subnet = micsPos(numMics+1:end,:);
        subscales = scales(2:end);
        trRTF = RTF_train(:,:,2:end);
    end
    if k == numArrays
        subnet = micsPos(1:end-numMics,:);
        subscales = scales(1:end-1);
        trRTF = RTF_train(:,:,1:end-1);
    end
    if and(k>1, k<numArrays)
       subnet = [micsPos(1:numMics*(k-1),:); micsPos(numMics*k+1:end,:)];
       subscales = [scales(1:k-1), scales(k+1:end)];
       trRTF = cat(3,RTF_train(:,:,1:k-1), RTF_train(:,:,k+1:end));
    end

end