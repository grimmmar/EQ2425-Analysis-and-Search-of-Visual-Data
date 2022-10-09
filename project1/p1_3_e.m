clc;
clear;

img = imread('data1\obj1_5.JPG');
img_gray = rgb2gray(img);
img_t = imread('data1\obj1_t1.jpg');
img_t_gray = rgb2gray(img_t);

strongest_thresh = 6000;
points = detectSURFFeatures(img_gray,'MetricThreshold',strongest_thresh);
[features,validPoints] = extractFeatures(img_gray, points);

points_t = detectSURFFeatures(img_t_gray,'MetricThreshold',strongest_thresh);
[features_t, validPoints_t] = extractFeatures(img_t_gray, points_t);

idx_min = 0;
matches = [];
thresh = 0.63;
features = features';
features_t = features_t';
pt_l = points.Location';
pt_l_t = points_t.Location';

for i = 1:size(features,2)
    dist_min = inf;
    dist_sec_min = inf;
    
    for j = 1:size(features_t,2)
        dist = sqrt(sum((features(:,i)-features_t(:,j)).^2));
        if dist < dist_min
            dist_min = dist;
            idx_min = j;
        elseif  (dist < dist_sec_min)&&(dist > dist_min)
            dist_sec_min = dist;
        end
    end
    ratio =  dist_min / dist_sec_min;
    if ratio < thresh
        matches = [matches;[i,idx_min]];
    end
end

imshow([img, img_t]);
hold on;

for i = 1:size(matches,1)
    x0 = pt_l(1,matches(i,1));
    y0 = pt_l(2,matches(i,1));
    x1 = pt_l_t(1,matches(i,2))+size(img,2) ;
    y1 = pt_l_t(2,matches(i,2));
    scatter(x0,y0,10,'r','filled');hold on;
    scatter(x1,y1,10,'b','filled');hold on; 
    line([x0,x1],[y0,y1],'color','yellow');
end
title('SURF Keypoints with Nearest Neighbor Distance Ratio Matching');