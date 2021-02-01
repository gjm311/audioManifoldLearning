% ---- load training data ----
load('mat_trainParams/biMicCircle_5L50U')
addpath ./functions
height = 1;
width = 4;
numRow = 10;
numCol = 40;
ref = [3,1,1];
varyMicPos = sourceGrid(height, width, numRow, numCol, ref);
movingArray = 1;
numMovePoints = 6;
height = 1;
init_pauses = 1;
end_pauses = 1;
movingMicsPos = [micLine(micsPos(movingArray,:), [.5,5.5,1], numMovePoints), height*ones(numMovePoints,1)];
varyMicPos = [[3 5 1].*ones(init_pauses,3);movingMicsPos];
% movingMicsPos = [movingMicsPos; [.5,5.5,1].*ones(end_pauses,3)];

micsPos = [3.025 5 1;2.975 5 1; 5 2.975 1;5 3.025 1; 3.025 1 1;2.975 1 1; 1 2.975 1; 1 2.975 1];
static_mics = subNetMinor(1, numArrays, numMics, micsPos);
min_right_mics = subNetMinor(2, numArrays, numMics, micsPos);
min_bottom_mics = subNetMinor(3, numArrays, numMics, micsPos);
min_left_mics = subNetMinor(4, numArrays, numMics, micsPos);

rtfDatabase = struct([]);
rtfDatabase(1).micsPos = micsPos;
rtfDatabase(1).staticFull = rtfEst(x, micsPos, rtfLen, numArrays, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
rtfDatabase(1).minTopSub = rtfEst(x, static_mics, rtfLen, numArrays-1, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
rtfDatabase(1).minRightSub = rtfEst(x, min_right_mics, rtfLen, numArrays-1, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
rtfDatabase(1).minBottomSub = rtfEst(x, min_bottom_mics, rtfLen, numArrays-1, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
rtfDatabase(1).minLeftSub = rtfEst(x, min_left_mics, rtfLen, numArrays-1, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);



for i = 2:size(varyMicPos,1)
    x1 = varyMicPos(i,1)-.025;
    x2 = varyMicPos(i,1)+.025;
    curr_varyMicPos = [x1, varyMicPos(i,2:3); x2, varyMicPos(i,2:3)];
    curr_micsPos = [curr_varyMicPos; micsPos(3:8,:)]; 
    min_right_mics = subNetMinor(2, numArrays, numMics, curr_micsPos);
    min_bottom_mics = subNetMinor(3, numArrays, numMics, curr_micsPos);
    min_left_mics = subNetMinor(4, numArrays, numMics, curr_micsPos);

    rtfDatabase(i).micsPos = curr_micsPos;
    rtfDatabase(i).minRightSub = rtfEst(x, min_right_mics, rtfLen, numArrays-1, numMics, sourceTrain, roomSize, T60, rirLen, c, fs);
    rtfDatabase(i).minBottomSub = rtfEst(x, min_bottom_mics, rtfLen, numArrays-1, numMics, sourceTrain, roomSize, T60, rirLen, c, fs); 
    rtfDatabase(i).minLeftSub = rtfEst(x, min_left_mics, rtfLen, numArrays-1, numMics, sourceTrain, roomSize, T60, rirLen, c, fs); 
end

%Is it possible to sim RTFs for test points?

save('mat_trainParams/rtfDatabase');