% ---- load training data ----
load('mat_trainParams/biMicCircle_gridL50U')
addpath ./functions
height = 1;
width = 4;
numRow = 10;
numCol = 40;
ref = [3,1,1];
varyMicPos = sourceGrid(height, width, numRow, numCol, ref);

rtfDatabase = struct([]);


for i = 1:size(varyMicPos,1)
    x1 = varyMicPos(i,1)-.1;
    x2 = varyMicPos(i,1)+.1;
    curr_varyMicPos = [x1, varyMicPos(i,2), varyMicPos(i,3); x2, varyMicPos(i,2), varyMicPos(i,3)];
    micsPos = [micsPos(1:4,:); curr_varyMicPos; micsPos(7:8,:)]; 
    rtfs(:,:,:,i) = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
    rtfDatabase(i).a = curr_varyMicPos;
    rtfDatabase(i).b = rtfs;
end

save('mat_trainParams/rtfDatabase');