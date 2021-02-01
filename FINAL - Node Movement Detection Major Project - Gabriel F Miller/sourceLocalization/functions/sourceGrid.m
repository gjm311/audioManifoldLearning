function sources = sourceGrid(height, width, numRow, numCol, ref)
    numRow = numRow-1;
    numCol = numCol-1;
    start_x = ref(1) - width/2;
    start_y = ref(2) - height/2;
    end_x = ref(1) + width/2;
    end_y =  ref(2) + height/2;
    x_lin = start_x:(end_x-start_x)/numCol:end_x;
    y_lin = start_y:(end_y-start_y)/numRow:end_y;
    
    sources = zeros(numRow*numCol,3);
    
    for x = 1:size(x_lin,2)
        for y = 1:size(y_lin,2)
            sources((x-1)*size(y_lin,2)+y,:) = [x_lin(x), y_lin(y),1];
        end
    end
end