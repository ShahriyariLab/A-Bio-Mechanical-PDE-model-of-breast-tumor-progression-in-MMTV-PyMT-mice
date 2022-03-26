%#########################################################################%
%This function outputs a set of points which form an ellipse and saves the
%points in a csv file for later use such as meshing and etc.
% (x,y): coordinates of the center
% a: The horizontal radius
% b: The vertical radius
% theta: Rotation angle
%Code by: Navid Mohammad Mirzaei (sites.google.com/view/nmirzaei)
%#########################################################################%
function h = ellipse(x,y,a,b,theta)
R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
t = linspace(0,2*pi,200) ;
xunit = a*cos(t);
yunit = b*sin(t);

A = R*[xunit;yunit]+[x;y];
writematrix(A,'EllipseBD.csv');
%#########################################################################%
