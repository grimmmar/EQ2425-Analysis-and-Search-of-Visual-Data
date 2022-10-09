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
pl = points.Location';
x_surf = [];
y_surf = [];
    

for i = 0:8
    
    scale = 1.2.^i;
    img_scale = imresize(img_gray_single, scale);
    matches = 0;  
    [f_scale,d_scale] = vl_sift(img_scale,'PeakThresh', peak_thresh, 'edgethresh', edge_thresh); 
    f_my_scale = f(1:2,:) * scale;
    
    for j = f_my_scale(1:2,:)
        for k = f_scale(1:2,:)
            if (abs(k(1)-j(1)) <= 2) && (abs(k(2)-j(2)) <= 2)
                matches = matches + 1;
                break
            end            
        end
    end

    repeatability = matches / size(f_my_scale,2);
    y = [y,repeatability];
    x = [x,scale];
    
    img_gray_scale = imresize(img_gray, scale);
    matches = 0;
    
    surf_scale = detectSURFFeatures(img_gray_scale,'MetricThreshold',strongest_thresh);
    points_scale = surf_scale.Location';    
    points_my_scale = pl(1:2,:) * scale;
    
    for j = points_my_scale(1:2,:)
        for k = points_scale(1:2,:)
            if (abs(k(1)-j(1)) <= 2) && (abs(k(2)-j(2)) <= 2)
                matches = matches + 1;
                break
            end            
        end
    end
    
    repeatability = matches / size(points_my_scale,2);
    y_surf = [y_surf,repeatability];
    x_surf = [x_surf,scale];   
    
end

plot(x,y,'red-*'); hold on;
plot(x_surf,y_surf,'blue-o');    

title('Repeatability vs Scaling factor');           
xlabel('Scaling factor');                    
ylabel('Repeatability');  
legend('SIFT','SURF')