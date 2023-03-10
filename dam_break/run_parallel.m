%% Runs flooding simulations at sites specified by site_data.csv in parallel
clear, clc, close all;
addpath('model');

%% Parameters
Nobj = 9000;
tmax = 2000;
startI = 7;

%% Data format: SiteName	Lat	Long	Volume(Mm^3)	Density (kg/m^3)	PondRadius(m)
siteData = importdata('site_data.csv');
N = size(siteData.data,1);

parfor i=startI:N
    siteName = siteData.textdata{i+1,1};
    siteLat = siteData.data(i,1);
    siteLong = siteData.data(i,2);
    vol = siteData.data(i,3)*1e6;
    rho = siteData.data(i,4);
    pondRadius = siteData.data(i,5);
    
    fprintf('\n Simulating: %s\n',siteName)
    main_func({siteName,siteLong,siteLat,pondRadius,Nobj,vol,rho,tmax});
    fprintf('\n Finished: %s...\n',siteName)
end

