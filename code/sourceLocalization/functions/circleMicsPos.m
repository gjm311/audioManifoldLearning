function micsPos = circleMicsPos(numArrays, numMics,focus,radiusL,radiusS,height)
    arraysPos = arrayMicsPos(numArrays, focus, radiusL);
    totalMics = numArrays*numMics;
    micsPos = zeros(totalMics,3);
    for i = 1:numArrays
%         micsPos(((i-1)*numMics)+1,:) = arraysPos(i,:);
        micsPos(((i-1)*numMics)+1:((i-1)*numMics)+numMics,:) = arrayMicsPos(numMics-1, arraysPos(i,:), radiusS);
    end
    micsPos(:,3) = micsPos(:,3)*height;

    function arraysPos = arrayMicsPos(numMics, focus, radius)
        x0=focus(1); % x0 and y0 are center coordinates
        y0=focus(2);  
        angle=0:2*pi/numMics:2*pi;
        angle = angle(2:end);
        x=radius*cos(angle)+x0;
        y=radius*sin(angle)+y0;
        arraysPos = [x(:), y(:), ones(size(x,2),1)];
    end

end
