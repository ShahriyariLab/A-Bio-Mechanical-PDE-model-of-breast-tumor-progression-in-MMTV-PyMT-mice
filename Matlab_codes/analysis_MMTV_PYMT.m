%#########################################################################%
%#########################################################################%
%This code generates spatial distribution of cells and biomarkers based on
%biological data reported from experiments on an MMTV-PyMT mouse model
%#########################################################################%
%#########################################################################%
%Original Code: Young Hwan Chang (sites.google.com/site/yhchangucb/)

%Modified by: Navid Mohammad Mirzaei (sites.google.com/view/nmirzaei)

%Corresponding paper:  "A Bio-Mechanical PDE model of breast tumor 
%progression in MMTV-PyMT mice"
%#########################################################################%
%#########################################################################%
clear all; clc; close all;
%#########################################################################%
%Reading the early and late fold-out data
%#########################################################################%
d_early = readtable('MMTV-PyMT mIHC early tumor fold out R101.csv');
d_late = readtable('MMTV-PyMT mIHC late tumor fold out R97.csv');
%#########################################################################%

%#########################################################################%
%Array of biomarkers' names
%#########################################################################%
feat_name = {'Arg1', 'aSMA', 'CC3', 'CD11b', 'CD11c', ...
              'CD31', 'CD3', 'CD45','CD4','CD8',...
              'CSF1R', 'Epcam', 'F480', 'Foxp3', 'Ki67', ...
              'Ly6G', 'MHCII', 'PDL1'};
%#########################################################################%

%#########################################################################%
%Late and early expression thresholds
%#########################################################################%
late_cell_num = [ 1313   233     13      388     152 ...
                  235    12      95      12      28 ...
                  74     10384   74      31      3344 ...
                  19     358     23      ];
              
early_cell_num = [  85   1194    49      454     108 ...
                    593  302    358      100     175 ...
                    379  9351   356      13     4046 ...
                    10   939    524];
%#########################################################################%                

%#########################################################################%
%Turning the data into arrays
%#########################################################################%
f_early = table2array(d_early(:,3:end-2));
pos_early = table2array(d_early(:,end-1:end));
f_late = table2array(d_late(:,3:end-2));
pos_late = table2array(d_late(:,end-1:end));
%#########################################################################%

%#########################################################################%
%Determining the boundary of data 
%#########################################################################%
k1 = boundary(pos_early(:,1)/10000,pos_early(:,2)/10000);
k2 = boundary(pos_late(:,1)/10000,pos_late(:,2)/10000);
%#########################################################################%

%#########################################################################%
%Ploting all the biomarkers along with an ellipsoid
%This ellipsoid was later used as the mathematical domain for PDE
%simulation
%#########################################################################%
figure; 
%division by 10000 is to turn the scales into cm
subplot(221); plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.','Color',...
    [0.5 0.5 0.5]); 
title('early');axis equal
ellipse(0.21,0.06,0.047,0.057,-pi/5)
subplot(222); plot(pos_late(:,1)/10000,pos_late(:,2)/10000,'.','Color',...
    [0.5 0.5 0.5]); 
title('late');axis equal
subplot(223); plot(pos_early(k1,1)/10000,pos_early(k1,2)/10000); 
title('early boundary');axis equal
subplot(224); plot(pos_late(k2,1)/10000,pos_late(k2,2)/10000); 
title('late boundary');axis equal
%#########################################################################%

%#########################################################################%
%write the boundary of the early fold out into a file
%#########################################################################%
B = [pos_early(k1,1)'/10000;pos_early(k1,2)'/10000];
D = [pos_early(:,1)'/10000;pos_early(:,2)'/10000];
writematrix(B,'Boundary.csv');
writematrix(D,'Domain.csv');
writematrix(k1','indecies.csv')
%#########################################################################%

%#########################################################################%
%Threshold quantiles
%#########################################################################%
for i=1:length(early_cell_num)
    Th_early(i) = quantile(f_early(:,i), 1-early_cell_num(i)/size(f_early,1));
end

for i=1:length(late_cell_num)
    Th_late(i) = quantile(f_late(:,i), 1-late_cell_num(i)/size(f_late,1));
end
%#########################################################################%

%#########################################################################%
%Plotting the thresolds
%#########################################################################%
r_cell = [early_cell_num./size(f_early,1) ; late_cell_num./size(f_late,1)];
figure
bar(r_cell','group');
xticks(1:length(feat_name));
xticklabels(feat_name);
xtickangle(90);
legend('early','late');
ylabel('Pos ratio');
%#########################################################################%

%#########################################################################%
%plotting single biomarkers for the early foldout
%#########################################################################%
figure('pos',[10 10 1000 600]);
for i=1:length(early_cell_num)
    subplot(4,5,i); 
    plot(pos_early(:,1),pos_early(:,2),'.', 'Color',[0.5 0.5 0.5]);
    hold on;
    ii = []; ii = find(f_early(:,i) > Th_early(i));
    plot(pos_early(ii,1),pos_early(ii,2),'.b');
    title(sprintf('%s - pos', feat_name{i}));
end
%#########################################################################%    

%#########################################################################%
%plotting single biomarkers for the late foldout
%#########################################################################%
figure('pos',[10 10 1000 600]);
for i=1:length(late_cell_num)
    subplot(4,5,i); 
    plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
    hold on;
    ii = []; ii = find(f_late(:,i) > Th_late(i));
    plot(pos_late(ii,1),pos_late(ii,2),'.r');
    title(sprintf('%s - pos', feat_name{i}));
end
%#########################################################################%
%plotting single biomarkers for the early foldout
%#########################################################################%

%#########################################################################%
%Determining the locations with different cell type expressions for the 
%early foldout
%#########################################################################%
ii_Th = []; 
ii_Th = f_early(:,12)<Th_early(12) & f_early(:,8)>Th_early(8) & ...
    f_early(:,7)>Th_early(7) & f_early(:,9)>Th_early(9) & ...
    f_early(:,10)<Th_early(10);

ii_Tc = []; 
ii_Tc = f_early(:,12)<Th_early(12) & f_early(:,8)>Th_early(8) & ...
    f_early(:,7)>Th_early(7) & f_early(:,9)<Th_early(9) & ...
    f_early(:,10)>Th_early(10);

ii_Tr = []; 
ii_Tr = f_early(:,12)<Th_early(12) & f_early(:,8)>Th_early(8) &...
    f_early(:,7)>Th_early(7) & f_early(:,9)<Th_early(9) & ...
    f_early(:,10)<Th_early(10) ;

ii_Dn = []; 
ii_Dn = f_early(:,12)<Th_early(12) & f_early(:,8)>Th_early(8) & ...
    f_early(:,13)<Th_early(13) & f_early(:,5)>Th_early(5);

ii_D = []; 
ii_D = f_early(:,12)<Th_early(12) & f_early(:,8)>Th_early(8) & ...
    f_early(:,13)<Th_early(13) & f_early(:,5)>Th_early(5) & ...
    f_early(:,17)>Th_early(17);

ii_M = []; 
ii_M = (f_early(:,12)<Th_early(12) & f_early(:,8)>Th_early(8) & ...
    f_early(:,13)>Th_early(13) & f_early(:,5)<Th_early(5) & ...
    f_early(:,11)>Th_early(11)) | (f_early(:,12)<Th_early(12) & ...
    f_early(:,8)>Th_early(8) & f_early(:,13)>Th_early(13) & ...
    f_early(:,5)<Th_early(5) & f_early(:,11)<Th_early(11) & ...
    f_early(:,17)>Th_early(17));

ii_C = []; 
ii_C = f_early(:,12)>Th_early(12) & f_early(:,8)<Th_early(8);

ii_N = []; 
ii_N = f_early(:,3)>Th_early(3);

ii_A = []; 
ii_A = f_early(:,12)<Th_early(12) & f_early(:,8)<Th_early(8);
%#########################################################################%

%#########################################################################%
%Plotting the locations of different cell types separately for the 
%early foldout along with the ellipsoid
%#########################################################################%
figure('pos',[10 10 1000 600]);
subplot(4,5,1); 
plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_early(ii_Th,1)/10000,pos_early(ii_Th,2)/10000,'.b');
title(sprintf('Th - pos'));
ellipse(0.21,0.06,0.047,0.057,-pi/5)
subplot(4,5,2); 
plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_early(ii_Tc,1)/10000,pos_early(ii_Tc,2)/10000,'.r');
title(sprintf('Tc - pos'));
ellipse(0.21,0.06,0.047,0.057,-pi/5)
subplot(4,5,3); 
plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_early(ii_Tr,1)/10000,pos_early(ii_Tr,2)/10000,'.g');
title(sprintf('Tr - pos'));
ellipse(0.21,0.06,0.047,0.057,-pi/5)
subplot(4,5,4); 
plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_early(ii_Dn,1)/10000,pos_early(ii_Dn,2)/10000,'.m');
title(sprintf('Dn - pos'));
ellipse(0.21,0.06,0.047,0.057,-pi/5)
subplot(4,5,5); 
plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_early(ii_D,1)/10000,pos_early(ii_D,2)/10000,'.black');
title(sprintf('D - pos'));
ellipse(0.21,0.06,0.047,0.057,-pi/5)
subplot(4,5,6); 
plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_early(ii_M,1)/10000,pos_early(ii_M,2)/10000,'.b');
title(sprintf('M - pos'));
ellipse(0.21,0.06,0.047,0.057,-pi/5)
subplot(4,5,7); 
plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_early(ii_C,1)/10000,pos_early(ii_C,2)/10000,'.y');
title(sprintf('C - pos'));
ellipse(0.21,0.06,0.047,0.057,-pi/5)
subplot(4,5,8); 
plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_early(ii_N,1)/10000,pos_early(ii_N,2)/10000,'.r');
title(sprintf('N - pos'));
ellipse(0.21,0.06,0.047,0.057,-pi/5)
subplot(4,5,9); 
plot(pos_early(:,1)/10000,pos_early(:,2)/10000,'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_early(ii_A,1)/10000,pos_early(ii_A,2)/10000,'.g');
title(sprintf('A - pos'));
ellipse(0.21,0.06,0.047,0.057,-pi/5)
%#########################################################################%

%#########################################################################%
%Plotting the all the cell types together for the early foldout along with 
%the ellipsoid
%#########################################################################%
figure
hold on
plot(pos_early(ii_Th,1)/10000,pos_early(ii_Th,2)/10000,'.b');
plot(pos_early(ii_Tc,1)/10000,pos_early(ii_Tc,2)/10000,'.r');
plot(pos_early(ii_Tr,1)/10000,pos_early(ii_Tr,2)/10000,'.g');
plot(pos_early(ii_Dn,1)/10000,pos_early(ii_Dn,2)/10000,'.m');
plot(pos_early(ii_D,1)/10000,pos_early(ii_D,2)/10000,'.black');
plot(pos_early(ii_M,1)/10000,pos_early(ii_M,2)/10000,'.b');
plot(pos_early(ii_C,1)/10000,pos_early(ii_C,2)/10000,'.y');
plot(pos_early(ii_N,1)/10000,pos_early(ii_N,2)/10000,'.r');
plot(pos_early(ii_A,1)/10000,pos_early(ii_A,2)/10000,'.g');
ellipse(0.21,0.06,0.047,0.057,-pi/5)
%#########################################################################%


%#########################################################################%
%Determining the locations with different cell type expressions for the 
%late foldout
%#########################################################################%
ii_Th = []; 
ii_Th = f_late(:,12)<Th_late(12) & f_late(:,8)>Th_late(8) & ...
    f_late(:,7)>Th_late(7) & f_late(:,9)>Th_late(9) & ...
    f_late(:,10)<Th_late(10);

ii_Tc = []; 
ii_Tc = f_late(:,12)<Th_late(12) & f_late(:,8)>Th_late(8) & ...
    f_late(:,7)>Th_late(7) & f_late(:,9)<Th_late(9) & ...
    f_late(:,10)>Th_late(10);

ii_Tr = []; 
ii_Tr = f_late(:,12)<Th_late(12) & f_late(:,8)>Th_late(8) & ...
    f_late(:,7)>Th_late(7) & f_late(:,9)>Th_late(9) & ...
    f_late(:,10)<Th_late(10) & f_late(:,14)>Th_late(14);

ii_Dn = []; 
ii_Dn = f_late(:,12)<Th_late(12) & f_late(:,8)>Th_late(8) & ...
    f_late(:,13)<Th_late(13) & f_late(:,5)>Th_late(5);

ii_D = []; 
ii_D = f_late(:,12)<Th_late(12) & f_late(:,8)>Th_late(8) & ...
    f_late(:,13)<Th_late(13) & f_late(:,5)>Th_late(5) & ...
    f_late(:,17)>Th_late(17);

ii_M = []; 
ii_M = (f_late(:,12)<Th_late(12) & f_late(:,8)>Th_late(8) & ...
    f_late(:,13)>Th_late(13) & f_late(:,5)<Th_late(5) & ...
    f_late(:,11)>Th_late(11)) | (f_late(:,12)<Th_late(12) & ...
    f_late(:,8)>Th_late(8) & f_late(:,13)>Th_late(13) & ...
    f_late(:,5)<Th_late(5) & f_late(:,11)<Th_late(11) & ...
    f_late(:,17)>Th_late(17));

ii_C = []; 
ii_C = f_late(:,12)>Th_late(12) & f_late(:,8)<Th_late(8);

ii_N = []; 
ii_N = f_late(:,3)>Th_late(3);

ii_A = []; 
ii_A = f_late(:,12)<Th_late(12) & f_late(:,8)<Th_late(8);
%#########################################################################%

%#########################################################################%
%Plotting the locations of different cell types separately for the 
%early foldout along with the ellipsoid
%#########################################################################%
figure('pos',[10 10 1000 600]);
subplot(4,5,1); 
plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_late(ii_Th,1),pos_late(ii_Th,2),'.b');
title(sprintf('Th - pos'));
subplot(4,5,2); 
plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_late(ii_Tc,1),pos_late(ii_Tc,2),'.r');
title(sprintf('Tc - pos'));
subplot(4,5,3); 
plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_late(ii_Tr,1),pos_late(ii_Tr,2),'.g');
title(sprintf('Tr - pos'));
subplot(4,5,4); 
plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_late(ii_Dn,1),pos_late(ii_Dn,2),'.m');
title(sprintf('Dn - pos'));
subplot(4,5,5); 
plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_late(ii_D,1),pos_late(ii_D,2),'.black');
title(sprintf('D - pos'));
subplot(4,5,6); 
plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_late(ii_M,1),pos_late(ii_M,2),'.b');
title(sprintf('M - pos'));
subplot(4,5,7); 
plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_late(ii_C,1),pos_late(ii_C,2),'.y');
title(sprintf('C - pos'));
subplot(4,5,8); 
plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_late(ii_N,1),pos_late(ii_N,2),'.r');
title(sprintf('N - pos'));
subplot(4,5,9); 
plot(pos_late(:,1),pos_late(:,2),'.', 'Color',[0.5 0.5 0.5]);
hold on;
plot(pos_late(ii_A,1),pos_late(ii_A,2),'.g');
title(sprintf('A - pos'));
%#########################################################################%