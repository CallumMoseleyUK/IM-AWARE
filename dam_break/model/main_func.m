function [simData] = main_func(inputList)
%(siteName, siteLong, siteLat, pondRadius, Nobj, vol, rho, tmax, c_visc, imaware_results_dir)

    siteName = inputList{1};
    siteLong = double(inputList{2});
    siteLat = double(inputList{3});
    pondRadius = double(inputList{4});
    Nobj = max(100,double(inputList{5}));
    vol = double(inputList{6});
    rho = double(inputList{7});
    tmax = double(inputList{8});
    c_visc = max(0,double(inputList{9}));
    imaware_results_dir = inputList{10};
    
    %% Simulation parameters
    % Number of discrete spheres along pond diameter
    %Nradial = 6;%12;
    % Height of the dam
    %dam_height= 80;
    % Tailings volume (released) (m^3)
    %vol = 9.57e6;
    % Tailings density (kg/m^3)
    %rho = 1594;
    % Time span to simulate (seconds)
    %tmax = 300;%20*60; %400*60;
    % Time step of simulation (seconds)
    dt = 0.2;
    % Iterations between collision calculations 
    collision_tick = 5;
    % Linear friction coefficient between each sphere and the terrain
    c = 1.0;
    % Rotational damping coefficient
    %c_visc = 0.04;
    % Collision friction
    c_col = 0.00;

    % Standard deviation as a ratio of the mean of particle radus
    r_sigma = 0.0;
    
    bScale = true;

    %% Import map data
    fprintf('\n Retrieving map\n');
    for i=1:10
        try
            [mapZ,fname,mapLat,mapLong] = get_map(siteLat,siteLong);
            break;
        catch error
            fprintf(error.identifier);
            pause(1.0);
        end
    end
    mapRes = 30;

    %% Find site location in pixels and meters
    [mapRows,mapCols] = size(mapZ);
    pxLat = 1 + (siteLat - mapLat(1)) * ((mapRows-1)/(mapLat(2)-mapLat(1)));
    pxLong = 1 + (siteLong - mapLong(1)) * ((mapCols-1)/(mapLong(2)-mapLong(1)));

    siteX = pxLong*mapRes;
    siteY = pxLat*mapRes;
    siteZ = double(mapZ(round(mapRows-pxLat),round(pxLong)));

    %% Run simulation
    s0 = [siteX;siteY;siteZ];
    v0 = [0;0;0];
    w0 = [0;0;0];

    fprintf('\nRunning model\n');
    tic;
    %[t,objects,data] = sim_dambreak_original_optimised(Nradial,tmax,dt,collision_tick,s0,v0,w0,mapRes,mapZ,pondRadius,vol,rho,c,c_visc,c_col);
    [t,objects,data] = sim_dambreak_stochastic(Nobj,tmax,dt,collision_tick,s0,v0,w0,mapRes,mapZ,pondRadius,vol,rho,c,c_visc,c_col,r_sigma);
    runtime = toc;

    %% Save results as .mat and .csv
    % NOTE: you don't need to keep .mat files, but they can be used to
    % regenerate .csv data if the csv data format needs to be retroactively
    % changed.
    %
    simID = siteName;
    [funcDir,~,~] = fileparts(mfilename('fullpath'));

    % Set csv file location (unique)
    sharedDir = split(funcDir,'SRC');
    sharedDir = sharedDir{1};
    
    %csvDir = [funcDir,'/results_csv'];
    %csvDir = [sharedDir,'IMAWARE/Analysis_Results'];
    csvDir = fullfile(sharedDir,imaware_results_dir);
    csvFile = fullfile(csvDir,sprintf('%s.csv',simID));
    if ~exist(csvDir,'dir')
        mkdir(csvDir);
    end
    i = 0;
    while exist(csvFile,'file')
        simID = sprintf('%s_%i',siteName,i);
        csvFile = fullfile(csvDir,sprintf('%s.csv',simID));
        i = i+1;
    end

    % Set mat file location (unique, based on csv)
    %resultsDir = [funcDir, '/results_mat'];
    resultsDir = fullfile(sharedDir, 'IMAWARE','Sim_Raw','results_mat');
    
    if ~exist(resultsDir,'dir')
        mkdir(resultsDir);
    end
    resultsFile = fullfile(resultsDir,sprintf('%s.mat',simID));

    % Save results file and convert to csv
    save(resultsFile);
    trajectory2csv_file(resultsFile,csvDir);
    
    %% Return sim data
    simData = {data.Nobj,data.m(1),data.r(1),tmax,c_visc,simID,csvFile};
    fprintf('[DAMBREAK_OUTPUT]%i,%.10f,%.10f,%.10f,%.10f,%s,%s',data.Nobj,data.m(1),data.r(1),tmax,c_visc,simID,csvFile);

end
function [csvName] = trajectory2csv_file(resultsFile,destFolder)
    %% Load object trajectories
    [fpath,fname,fext] = fileparts(resultsFile);
    csvName = sprintf('%s/%s.csv',destFolder,fname);
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