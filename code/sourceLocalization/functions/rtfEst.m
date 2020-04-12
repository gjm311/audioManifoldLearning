function rtfs = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourcePos, roomSize, T60, rirLen, c, fs)
    numSources = size(sourcePos,1);
    rtfs = zeros(numSources, rtfLen, numArrays);
    maxDelay = ceil(norm(micsPos(1,:)-micsPos(2,:))/c*fs);
    
    if numArrays == 1
        na = numMics;
    else
        na = numArrays;
    end

    for j = 1:na
        for t = 1:numSources
            if or(numArrays > 1, j == 1)
                currMics = micsPos((j-1)*numMics+1:j*numMics,:);
                if size(x,1) > 1
                    x_curr = x(j,:);
                else
                    x_curr = x;
                end
                curr = [currMics(1,:); currMics(2,:)];    
                RIRs = rir_generator(c, fs, curr, sourcePos(t,:), roomSize,...
                    T60, rirLen);

                y1 = conv(x_curr, RIRs(1,:))';
                y2 = conv(x_curr, RIRs(2,:))';
                %RTF_train(i,:) = TDRTF(rtfLen, y1, y2, maxDelay);
                [rtfs(t,:,j),~,~] = FDRTF(rtfLen, y1, y2, maxDelay);
            else
                if size(x,1) > 1
                    x_curr = x(j,:);
                else
                    x_curr = x;
                end
                curr = [micsPos(2,:); micsPos(1,:)];    
                RIRs = rir_generator(c, fs, curr, sourcePos(t,:), roomSize,...
                    T60, rirLen);

                y1 = conv(x_curr, RIRs(1,:))';
                y2 = conv(x_curr, RIRs(2,:))';
                %RTF_train(i,:) = TDRTF(rtfLen, y1, y2, maxDelay);
                [rtfs(t,:,j),~,~] = FDRTF(rtfLen, y1, y2, maxDelay);
                    
            end
                
            
        end
    end
%     normalize energies to 1
    if numSources > 1
        rtfs = sqrt(1./sum(rtfs,2)).*rtfs;
    end
%     if numArrays >1
%         rtfs = reshape(mean(rtfs), [numSources,rtfLen,numArrays]);
%     else
%         rtfs = reshape(rtfs, [numSources,rtfLen,numMics]);
%     end
end
