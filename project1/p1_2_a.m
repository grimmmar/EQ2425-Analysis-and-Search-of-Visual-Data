clc;
clear;

img = imread('data1\obj1_5.jpg');
img_gray_single = single(rgb2gray(img));
figure(1);
imshow(img); hold on;

peak_thresh = 13;
edge_thresh = 5;
[f,d] = vl_sift(img_gray_single,'PeakThresh', peak_thresh, 'edgethresh', edge_thresh);

sift = vl_plotframe(f) ;
set(sift,'color','r');

img_gray = rgb2gray(img);
strongest_thresh = 6000;
points = detectSURFFeatures(img_gray,'MetricThreshold',strongest_thresh);
pt_l = points.Location;
pt_l = double(transpose(pt_l));

figure(2);
imshow(img); hold on;
surf = vl_plotframe(pt_l) ;
set(surf,'color','g');