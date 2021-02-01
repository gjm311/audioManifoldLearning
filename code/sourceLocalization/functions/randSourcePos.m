function sourcePos = randSourcePos(numRand, roomSize, radius, reference)
    if radius > min(roomSize(:,1:2))
        sourcePos = rand(numRand, 3);
        sourcePos(:,1) = sourcePos(:,1)*roomSize(:,1);
        sourcePos(:,2) = sourcePos(:,2)*roomSize(:,2);
        sourcePos(:,3) = ones(numRand,1)*reference(3);
    else
        x1=roomSize(1)+1;
        y1=roomSize(2)+1;
        while(x1>roomSize(1) || y1>roomSize(2))
            x1 = reference(1);
            y1 = reference(2);
            z1 = reference(3);
            a=2*pi*rand(numRand,1);
            r=sqrt(rand(numRand,1));
            sourcePos = [(radius*r).*cos(a)+x1 (radius*r).*sin(a)+y1 ones(numRand,1)*z1];
    %         sourcePos(:,3) = ones(numRand,1)*reference(3);
        end
    end
end