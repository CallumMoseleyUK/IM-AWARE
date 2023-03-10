function [] = plotpaths(resultsFile,siteName)

    load(resultsFile);

    timeScale = 0.01;
    frameStep = 10;
    viewang = [0,90];
    viewrate = [0,0];
    viewsmoothing = 1;
    bScale = true;

    plotUnits = 1000;

    %% Render map (all this can be simplified by adapting render_results8.m and writing a function just for organising map dimensions)
    fig = figure('visible','off');

    fig.Position = [400,0,1024,1024];
    [mapRows,mapCols] = size(mapZ);
    mapX = ((1:mapCols)-1)*mapRes;
    mapY = ((1:mapRows)-1)*mapRes;

    % Get axis scale
    if bScale
        xlim = [Inf,0];
        ylim = xlim;
        for k=1:length(objects)
            x = objects(k).x(:,1);
            y = objects(k).x(:,2);
            z = objects(k).x(:,3);

            xlim(1) = min(xlim(1),min(x));
            xlim(2) = max(xlim(2),max(x));
            ylim(1) = min(ylim(1),min(y));
            ylim(2) = max(ylim(2),max(y));

        end
        %xlim(1) = xlim(1)*0.9;
        %xlim(2) = xlim(2)*1.1;
        %ylim(1) = ylim(1)*1.1;
        %ylim(1) = ylim(2)*0.9;
	meanX = mean(xlim);
	meanY = mean(ylim);
	dx = abs(diff(xlim))/2;
	dy = abs(diff(ylim))/2;
	dx = max(dx,dy);
	dy = max(dy,dx);
	xlim = [meanX-dx*2,meanX+dx*2];
	ylim = [meanY-dy*2,meanY+dy*2];
        
    else
        xlim = [min(mapX(:)),max(mapX(:))];
        ylim = [min(mapY(:)),max(mapY(:))];
    end

    % --- make square
%         xlim(1) = min(xlim(1),ylim(1));
%         xlim(2) = max(xlim(2),ylim(2));
%         ylim = xlim;
    % 

    I = mapX>=xlim(1) & mapX<=xlim(2);
    J = mapY>=ylim(1) & mapY<=ylim(2);
    % --- get this working with plothmap.m
    %     [mapRows,mapCols] = size(mapZ);
    %     map_lim = mapRes*[0,mapRows-1,0,mapCols-1];
    % ----
    renderMap = flipud(mapZ);
    renderMap = double(renderMap(J,I));
    renderMap = renderMap - min(renderMap(:));
    renderMap = renderMap ./ max(renderMap(:));
    [lat,long] = pos2LatLong(mapX(I),mapY(J),mapLat,mapLong,mapRes,mapRows,mapCols);
    %image(long,lat,repmat(flipud(renderMap),1,1,3));
    image(long,lat,repmat(renderMap,1,1,3));
    
    ax = gca;
    ax.YDir = 'normal'; %image() reverses the y axis direction
    axis equal;
    %axis_lim = axis;
    %axis([axis_lim(1:5),axis_lim(6)*1.5]);
    %set(gca,'cameraviewanglemode','manual');
    xlabel('West-East (deg)');
    ylabel('North-South (deg)');
    set(gca,'FontSize',20);
    view(viewang);


    %% Compare cases

    clearvars -except plotUnits fig siteName resultsFile mapRows mapCols; %clear data used to scale the map
    axis_lim = axis;

    contour_times = [300];
    shrinkFactor = [0.9,0.7,1];%[0.7,0.7,0.7];
    bSmooth = 0;
    
    fnames = {resultsFile};
    
    hold all;
    %leg = cell(length(stdev_pct),length(contour_times));
    flood_area = zeros(length(fnames),length(contour_times));
    for k=1:length(fnames)
        results = load(fnames{k});
        Nobj = length(results.objects);
        Nt = length(results.t);

        xpos = zeros(length(results.objects),1);
        ypos = xpos;
        upos = xpos;
        vpos = xpos;

        siteS = [results.siteX,results.siteY];

        s = zeros(Nt,1);
        for j=1:length(contour_times)

            %I = find(results.t==contour_times(j));

            for i=1:Nobj

                statek = results.objects(i).x(:,:);

                xpos(i) = statek(1);
                ypos(i) = statek(2);
                upos(i) = statek(4);
                vpos(i) = statek(5);

                [lat,long] = pos2LatLong(statek(:,1),statek(:,2),results.mapLat,results.mapLong,results.mapRes,mapRows,mapCols);
                plot(long,lat,'b-','LineWidth',2.5);
            end

            %zb = xb*0 + 2000;

            % Plot polygon
            %plot3(xb/plotUnits,yb/plotUnits,zb,'-','LineWidth',1.3);
            %leg{k,j} = sprintf('stdev=%i%%, time=%i',int16(stdev_pct(k)*100),contour_times(j));
        end

    end
    set(gca,'FontSize',20);
    
    [lat,long] = pos2LatLong(results.siteX,results.siteY,results.mapLat,results.mapLong,results.mapRes,mapRows,mapCols);
    plot(long,lat,'rx','MarkerSize',15,'LineWidth',1.8);
%     leg = [{''}; leg(:)];
%     legend(leg,'Location','BestOutside');

    axis(axis_lim);
    %legend(lbls,'Location','Best');
    fig.Position = [250 285 1059 693];
    if ~exist('static','dir')
        mkdir('static');
    end
    saveas(fig,'static/render_path.png');
    
    if ~exist('path_renders','dir')
        mkdir('path_renders');
    end
    saveas(fig,sprintf('path_renders/%s.png',siteName));
end

function [lat,long] = pos2LatLong(x,y,latRange,longRange,res,mapRows,mapCols)
    xRange = [0,res*mapCols];
    yRange = [0,res*mapRows];
    long = min(longRange) + (max(longRange)-min(longRange)) * (x-min(xRange))./(max(xRange)-min(xRange));
    lat = min(latRange) + (max(latRange)-min(latRange)) * (y-min(yRange))./(max(yRange)-min(yRange));
end
