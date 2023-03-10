function [mapZ,tifDir,mapLat,mapLong] = get_map(siteLat,siteLong)

    %dataDir = 'data_DEM';
    %dataDir = sprintf('%s/%s',pwd,'data_DEM');
    %[dataDir,~,~] = fileparts(mfilename('fullpath'));
    %dataDir = [dataDir, '/data_DEM'];
    
    sharedDir = split(pwd,'SRC');
    dataDir = fullfile(sharedDir{1},'IMAWARE','Sim_Raw','data_DEM');
    
    %% Generate tile code
    % Negative latitude indicates position measured toward south, and negative
    % longitude means position measured toward west.
    % Tile code format e.g. S021W045, which covers the area between 21 degrees
    % south, 45 degrees west to 20 degrees south, 44 degrees south. Hence,
    % values in the code should be rounded no matter what (negatives become more negative and positives become smaller).
    % Positions in the tile code should always be three characters long,
    % appending 0 to the beginning if the value is fewer than 3 digits long.
    if siteLat<0
        tileCodeLat = sprintf('S%.3i',abs(floor(siteLat)));
        mapCodeLat = sprintf('S%.3i',abs(floor(siteLat/5)*5));
    else
        tileCodeLat = sprintf('N%.3i',abs(floor(siteLat)));
        mapCodeLat = sprintf('N%.3i',abs(floor(siteLat/5)*5));
    end
    if siteLong<0
        tileCodeLong = sprintf('W%.3i',abs(floor(siteLong)));
        mapCodeLong = sprintf('W%.3i',abs(floor(siteLong/5)*5));
    else
        tileCodeLong = sprintf('E%.3i',abs(floor(siteLong)));
        mapCodeLong = sprintf('E%.3i',abs(floor(siteLong/5)*5));
    end
    tileCode = [tileCodeLat,tileCodeLong];
    mapCode = [mapCodeLat,mapCodeLong];

    %% Generate URLs, filepaths, etc
    fname = sprintf('%s.zip',tileCode);
    url = sprintf('https://www.eorc.jaxa.jp/ALOS/aw3d30/data/release_v2012/%s/%s',mapCode,fname);
    tifFile = sprintf('ALPSMLC30_%s_DSM.tif',tileCode);
    tifDir = sprintf('%s/%s/%s',dataDir,tileCode,tifFile);

    %% Download and unzip file if not already downloaded. Otherwise, load the file.
    %options = weboptions('Username',username,'Password',password);
    if ~exist(tifDir,'file')
        unzip(url,dataDir);
    end

    mapZ = int16(importdata(tifDir));
    
    %% Calculate the map extent in degrees
    mapLat = [floor(siteLat),ceil(siteLat)];
    mapLong = [floor(siteLong),ceil(siteLong)];
end