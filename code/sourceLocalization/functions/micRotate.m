function micsPosNew = micRotate(roomSize, radius_mic, mic_ref, movingArray, micsPos, numArrays, numMics)
    randMicPos = randMicsPos(1, roomSize, radius_mic, mic_ref(movingArray,:));
    rotation = rand*360;
    new_x1 = randMicPos(:,1)+.025*cos(rotation);
    new_y1 = randMicPos(:,2)+.025*sin(rotation);
    new_x2 = randMicPos(:,1)-.025*cos(rotation);
    new_y2 = randMicPos(:,2)-.025*sin(rotation);
    if movingArray == 1
        micsPosNew = [new_x2 new_y2 1;new_x1 new_y1 1; micsPos(1+numMics:end,:)];
    elseif movingArray == numArrays
        micsPosNew = [micsPos(1:end-numMics,:); new_x2 new_y2 1; new_x1 new_y1 1];
    elseif movingArray == 2
        micsPosNew = [micsPos(1:numMics,:); new_x2 new_y2 1; new_x1 new_y1 1; micsPos(2*numMics+1:end,:)];
    else
        micsPosNew = [micsPos(1:numMics*2,:); new_x2 new_y2 1; new_x1 new_y1 1; micsPos(3*numMics+1:end,:)];
    end

end