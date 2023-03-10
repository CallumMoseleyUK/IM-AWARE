clear, clc, close all
addpath('model')

siteLong = -44.122;
siteLat = -20.122;
pondRadius = 300;
Nobj = 1000;
vol = 9.57e6;
rho = 1594;
tmax = 3000;
siteName = 'testSite_%i';

c_visc = linspace(0.001,0.04,5);
runTime = zeros(size(c_visc));
resultsFile = cell(size(c_visc));

c1 = rgb2hsv([1,0,0]);
c2 = rgb2hsv([0,0,1]);
col = linspace(c1(1),c2(1),length(c_visc));
cSat = 1.0;
cVal = 1.0;

delete(gcp('nocreate'));
parpool(3);
parfor i=1:length(c_visc)
    tic;
    siteNameI = sprintf(siteName,i);
    funcOut = main_func_visc({siteNameI,siteLong,siteLat,pondRadius,Nobj,vol,rho,tmax,c_visc(i)});
    runtime(i) = toc;
    
    resultsFile{i} = funcOut{1};
end

for i=1:length(runtime)
    fprintf('Run time: %1.4f\n',runtime(i));
    plotCode{i} = hsv2rgb([col(i),cSat,cVal]);
    leg{i} = sprintf('c_{v}=%1.3f',c_visc(i));
end

fig = figure('visible','on');
fig.Position = [400,0,1024,1024];
plotpaths2(fig,plotCode,resultsFile,siteName,leg);
saveas(fig,'visc_comparison.png');