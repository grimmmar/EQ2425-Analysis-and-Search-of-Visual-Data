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

figure(1);
imshow([img,img_t]); 
hold on;
sift = vl_plotframe(f);
set(sift,'color','r');
f_t(1,:) = f_t(1,:) + size(img_t,2);
sift_t = vl_plotframe(f_t);
set(sift_t,'color','g');
hold on;
legend('obj1\_5.JPG','obj1\_t1.JPG');
title('SIFT Keypoints');
