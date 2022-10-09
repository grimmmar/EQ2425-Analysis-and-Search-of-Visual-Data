function [f_my_rotate] = my_rotate(width, height, f, angle)   

    f_my_rotate = f; 
    
    for i = 1:size(f,2)
        width_n = abs(width * cosd(angle)) + abs(height * sind(angle));
        height_n = abs(width * sind(angle)) + abs(height * cosd(angle));
        
        matrix = [cosd(angle), -sind(angle);sind(angle),cosd(angle)]*[(f(1,i)-0.5*width);(0.5*height-f(2,i))];
        
        f_my_rotate(1,i) = matrix(1,1) + 0.5 * width_n;
        f_my_rotate(2,i) = 0.5 * height_n - matrix(2,1);
    end  
end