clc;
clear;

img = imread('data1\obj1_5.JPG');
img_gray_single = single(rgb2gray(img));

peak_thresh = 13;
edge_thresh = 5;
[f,d] = vl_sift(img_gray_single,'PeakThresh', peak_thresh, 'edgethresh', edge_thresh);
x = [];
y = [];
    
img_gray = rgb2gray(img);
strongest_thresh = 6000;
points = detectSURFFeatures(img_gray,'MetricThreshold',strongest_thresh);
points_location = points.Location';
x_surf = [];
y_surf = [];

for angle = 0:15:360
    img_rotate = imrotate(img_gray_single, angle);
    matches = 0;
    [f_rotate,d_rotate] = vl_sift(img_rotate,'PeakThresh', peak_thresh, 'edgethresh', edge_thresh);
    f_my_rotate = my_rotate(size(img,2),size(img,1),f,angle);
    
    for j = f_my_rotate(1:2,:)
        for k = f_rotate(1:2,:)
            if (abs(k(1)-j(1)) <= 2) && (abs(k(2)-j(2)) <= 2)
                matches = matches + 1;
                break
            end            
        end
    end
    
    repeatability = matches / size(f_my_rotate,2);
    x = [x, angle];
    y = [y, repeatability];
    
    
    img_gray_rotate = imrotate(img_gray, angle);
    surf_rotate = detectSURFFeatures(img_gray_rotate,'MetricThreshold',strongest_thresh);
    points_rotate = surf_rotate.Location';

    points_my_rotate = my_rotate(size(img,2),size(img,1),points_location,angle);
    
    matches = 0;

    for j = points_my_rotate(1:2,:)
        for k = points_rotate(1:2,:)
            if (abs(k(1)-j(1)) <= 2) && (abs(k(2)-j(2)) <= 2)
                matches = matches + 1;
                break
            end            
        end
    end
    
    repeatability = matches / size(points_my_rotate,2);
    x_surf = [x_surf,angle]; 
    y_surf = [y_surf,repeatability];       
end

plot(x,y,'red-*'); hold on;
plot(x_surf,y_surf,'blue-o');    

title('Repeatability vs Rotation Angle');           
xlabel('Rotation Angle');                    
ylabel('Repeatability');  
legend('SIFT','SURF')