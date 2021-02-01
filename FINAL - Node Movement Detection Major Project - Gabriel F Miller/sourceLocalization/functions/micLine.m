function vals = micLine(start, finish, points)
    if and(start(1) ~= finish(1),start(2) ~= finish(2)) 
        start_x = start(1);
        start_y = start(2);
        finish_x = finish(1);
        finish_y = finish(2);

        x_vals = [start_x];
        y_vals = [start_y];     

        x_len = finish_x - start_x;
        y_len = finish_y - start_y;      

        if start_x == finish_x
            for i = 1:points-1
                current_x = start_x;
                x_vals = [x_vals current_x];
                current_y = (y_len/(points-1))*i + start_y;
                y_vals = [y_vals current_y];     
            end
           vals = [x_vals; y_vals];
           return
        end

        if start_y == finish_y
            for i = 1:points-1
                current_y = start_y;
                y_vals = [y_vals current_y];
                current_x = (x_len/(points-1))*i + start_x;
                x_vals = [x_vals current_x];    
            end
            vals = [x_vals; y_vals];
            return
        end  
        slope = (finish_y - start_y)/(finish_x - start_x);
        y_int = start_y - start_x*slope; 
        for i = 1:points-1
            current_x = (x_len/(points-1))*i + start_x;
            x_vals = [x_vals current_x];
            current_y = slope*current_x + y_int;
            y_vals = [y_vals current_y];     
        end 
        vals = [x_vals; y_vals]';
    else
        vals = ones(points,2).*start(:,1:2);
    end
    
end