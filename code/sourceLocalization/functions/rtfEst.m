function rtfs = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourcePos, roomSize, T60, rirLen, c, fs)
    numRtfs = numMics*(numMics-1);
    numSources = size(sourcePos,1);
    rtfs = zeros(numRtfs, numSources, rtfLen, numArrays);
    maxDelay = ceil(max(pdist(micsPos))/c*fs);
    for j = 1:numArrays
        for t = 1:numSources
            currMics = micsPos((j-1)*numMics+1:j*numMics,:);
            if size(x,1) > 1
                x_curr = x(j,:);
            else
                x_curr = x;
            end
                
            
            iter = 1;
            for i = 1:numMics
                for k = 1:numMics
                    if i ~= k
                        curr = [currMics(i,:); currMics(k,:)];      
                        RIRs = rir_generator(c, fs, curr, sourcePos(t,:), roomSize,...
                            T60, rirLen);

                        y1 = conv(x_curr, RIRs(1,:))';
                        y2 = conv(x_curr, RIRs(2,:))';

                        %RTF_train(i,:) = TDRTF(rtfLen, y1, y2, maxDelay);
                        [rtfs(iter,t,:,j),~,~] = FDRTF(rtfLen, y1, y2, maxDelay);
                        iter = iter+1;
                    end
                end
            end
        end
    end
    %normalize energies to 1
    if numSources > 1
        rtfs = sqrt(1./sum(rtfs.^2,2)).*rtfs;
    end
    rtfs = reshape(mean(rtfs), [numSources,rtfLen,numArrays]);
end