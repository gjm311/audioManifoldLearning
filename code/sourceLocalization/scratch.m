movingMicsPos = zeros(18,3);
cnt = 1;
load('./mat_trainParams/rtfDatabase.mat', 'rtfDatabase');
for i = 153:-1:145
   currMics = rtfDatabase(i).micPos;
   movingMicsPos(cnt:cnt+1,:) = currMics;
   cnt = cnt +2;
end

