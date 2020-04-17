function micPos = randMicsPos(numRand, roomSize, radius, reference)    
    x1 = reference(1);
    y1 = reference(2);
    z1 = reference(3);
    rnd = rand(numRand,1)*pi;
    a=2*rnd*radius;
    r=sqrt(radius);
    micPos = [(radius*r).*cos(a)+x1 (radius*r).*sin(a)+y1 ones(numRand,1)*z1];
    
%     if micPos(1) > roomSize(1)
%        micPos = [reference(1)-radius 0 0]+micPos;
%     elseif micPos(1) < roomSize(1)
%         micPos(1) = [reference+radius 0 0]+micPos;
%     end
%     if micPos(2) > roomSize(2)
%        micPos(2) = reference(2)-radius;
%     elseif micPos(2) < roomSize(2)
%         micPos(2) = reference+radius;
%     end
end