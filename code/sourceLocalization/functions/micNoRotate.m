function [rotation,micsPosNew] = micNoRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics)
    randMicPos = randMicsPos(1, roomSize, radius_mic, mic_ref(movingArray,:));
    rotation = rand*360;
    
    if randMicPos == micsPos
        micsPosNew = micsPos;
    elseif movingArray == 1
        new_x1 = randMicPos(:,1)+.025;
        new_y1 = randMicPos(:,2);
        new_x2 = randMicPos(:,1)-.025;
        new_y2 = randMicPos(:,2);
        micsPosNew = [new_x1 new_y1 1;new_x2 new_y2 1; micsPos(1+numMics:end,:)];
    elseif movingArray == numArrays
        new_x1 = randMicPos(:,1);
        new_y1 = randMicPos(:,2)+.025;
        new_x2 = randMicPos(:,1);
        new_y2 = randMicPos(:,2)-.025;
        micsPosNew = [micsPos(1:end-numMics,:); new_x2 new_y2 1; new_x1 new_y1 1];
    elseif movingArray == 2
        new_x1 = randMicPos(:,1);
        new_y1 = randMicPos(:,2)+.025;
        new_x2 = randMicPos(:,1);
        new_y2 = randMicPos(:,2)-.025;
        micsPosNew = [micsPos(1:numMics,:); new_x2 new_y2 1; new_x1 new_y1 1; micsPos(2*numMics+1:end,:)];
    else
        new_x1 = randMicPos(:,1)+.025;
        new_y1 = randMicPos(:,2);
        new_x2 = randMicPos(:,1)-.025;
        new_y2 = randMicPos(:,2);
        micsPosNew = [micsPos(1:numMics*2,:); new_x1 new_y1 1; new_x2 new_y2 1; micsPos(3*numMics+1:end,:)];
    end

end