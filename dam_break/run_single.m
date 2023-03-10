%% Test script to extend flood distance with fewer particles

clear, clc, close all;
addpath('model');

siteLong =  -43.95436111111111;
siteLat = -20.04761111111111;
pondRadius = 196.4147936651236;
Nobj = 50;
vol =  2685782.0;
rho = 1594.0;
tmax = 30;
siteName = 'B3B4_Minerax-DAMBREAK-20210826-183352';
cVisc = 0.1;

fprintf('\n Simulating: %s\n',siteName)
tic
main_func({siteName,siteLong,siteLat,pondRadius,Nobj,vol,rho,tmax,cVisc,'IMAWARE/Analysis_Results'});
fprintf('\n Finished: %s...\n',siteName)
runtime = toc


% ['B3B4_Minerax-DAMBREAK-20210826-183352',
%  'B3B4_Minerax',
%  -20.04761111111111,
%  -43.95436111111111,
%  196.4147936651236,
%  50,
%  2685782.0,
%  1594.0,
%  30.0,
%  0.1,
%  0.0,
%  0.0,
%  1.0,
%  False]