function sourcePos = randSourcePos(numRand, roomSize, radius, reference)
    if radius > min(roomSize(:,1:2))
        sourcePos = rand(numRand, 3);
        sourcePos(:,1) = sourcePos(:,1)*roomSize(:,1);
        sourcePos(:,2) = sourcePos(:,2)*roomSize(:,2);
        sourcePos(:,3) = ones(numRand,1)*roomSize(3);
    else
        sourcePos = reference + (rand(numRand, 3)-.5)*radius;
        sourcePos(:,3) = ones(numRand,1)*roomSize(3);
    end
end