function micPos = randMicsPos(numRand, roomSize, radius, reference)
    rand_angle = rand*360;
    new_x = reference(:,1)+radius*cos(rand_angle);
    new_y = reference(:,2)+radius*sin(rand_angle);
    micPos = [new_x new_y reference(:,3)];
end