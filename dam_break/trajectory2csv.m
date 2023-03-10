%% Writes particle trajectories for a given set of results.mat files to csv format
% 
clear, clc, close all;

%srcFolder = [pwd,'/model/results_mat'];
sharedFolder = strsplit(pwd,'SRC');
srcFolder = fullfile(sharedFolder{1},'IMAWARE','Sim_Raw','results_mat');
destFolder = fullfile(sharedFolder{1},'IMAWARE','Analysis_Results');

trajectory2csv_folder(srcFolder,destFolder);


function [] = trajectory2csv_folder(srcFolder,destFolder)
    
    if ~exist(destFolder,'dir')
        mkdir(destFolder)
    end
    
    d = dir(srcFolder);
    parfor i=1:length(d)
        if d(i).isdir
            continue;
        end
        resultsFile = d(i).name;
        
        subFolder = strsplit(resultsFile,'-DAMBREAK');
        subFolder = subFolder{1};
        csvDir = [destFolder,'/',subFolder];
        
        %csvName = trajectory2csv_file([srcFolder, '/', d(i).name],destFolder);
        csvName = trajectory2csv_file([srcFolder,'/',resultsFile],csvDir);
        
    end
end
function [csvName] = trajectory2csv_file(resultsFile,destFolder)
    %% Load object trajectories
    [fpath,fname,fext] = fileparts(resultsFile);
    csvName = fullfile(destFolder,sprintf('%s.csv',fname));
    fprintf('Writing %s\nto %s\n', resultsFile,csvName);
    
    r = load(resultsFile);
    objects = r.objects;
    latRange = r.mapLat;
    longRange = r.mapLong;
    res = r.mapRes;
    t = r.t;
    [mapRows,mapCols] = size(r.mapZ);
    clear r;

    %% Store lat/long time histories
    Nt = length(t);
    Nobj = length(objects);
    data = [];
    for i=1:Nobj
        x = objects(i).x(:,1);
        y = objects(i).x(:,2);
        z = objects(i).x(:,3);
        vx = objects(i).x(:,4);
        vy = objects(i).x(:,5);
        vz = objects(i).x(:,6);
        
        [lat,long] = pos2LatLong(x,y,latRange,longRange,res,mapRows,mapCols);
        data = [data, [lat,long,z,vx,vy,vz]];
        %data = [data, [lat,long]];
        
    end
    % Prepend time vector to data
    data = [t,data];
    if ~exist(destFolder,'dir')
        mkdir(destFolder);
    end
    dlmwrite(csvName,data,'delimiter',',','precision',9);


end
function [lat,long] = pos2LatLong(x,y,latRange,longRange,res,mapRows,mapCols)
    xRange = [0,res*mapCols];
    yRange = [0,res*mapRows];
    long = min(longRange) + (max(longRange)-min(longRange)) * (x-min(xRange))./(max(xRange)-min(xRange));
    lat = min(latRange) + (max(latRange)-min(latRange)) * (y-min(yRange))./(max(yRange)-min(yRange));
end