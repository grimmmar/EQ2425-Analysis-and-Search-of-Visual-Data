clc;
clear;

img = imread('data1\obj1_5.JPG');
img_gray = single(rgb2gray(img));
img_t = imread('data1\obj1_t1.jpg');
img_t_gray = single(rgb2gray(img_t));

peak_thresh = 13;
edge_thresh = 5;

[f,d] = vl_sift(img_gray,'PeakThresh', peak_thresh, 'edgethresh', edge_thresh);
[f_t,d_t] = vl_sift(img_t_gray,'PeakThresh', peak_thresh, 'edgethresh', edge_thresh);

idx_min = 0;
matches = [];
thresh = 0.80;

for i = 1:size(d,2)
    dist_min = inf;
    dist_sec_min = inf;
    
    for j = 1:size(d_t,2)
        dist = sqrt(sum((d(:,i)-d_t(:,j)).^2));
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

imshow([img,img_t]);
hold on;

for i = 1:size(matches,1)
    x0 = f(1,matches(i,1));
    y0 = f(2,matches(i,1));
    x1 = f_t(1,matches(i,2))+size(img,2);
    y1 = f_t(2,matches(i,2));
    scatter(x0,y0,10,'r','filled');hold on;
    scatter(x1,y1,10,'b','filled');hold on; 
    line([x0,x1],[y0,y1],'color','yellow');
end
title('SIFT Keypoints with Nearest Neighbor Distance Ratio Matching');