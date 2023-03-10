function plot_spatial(resultsFile)
    load(resultsFile);

    timeScale = 0.01;
    frameStep = 10;
    bMakeGif = true;
    gifname = 'static/animated3d.gif';
    viewang = [-30,30];
    viewrate = [0,0];
    viewsmoothing = 1;
    bScale = true;

    %% Check existing files
%     if bMakeGif
%         for i=0:9
%             fn = sprintf(gifname,i);
%             if ~exist(fn)
%                 gifname = fn;
%                 break;
%             end
%         end
%     end

    %% Render
    fig = figure('visible','off');

    fig.Position = [400,0,1024,1024];
    [mapRows,mapCols] = size(mapZ);
    mapX = ((1:mapCols)-1)*mapRes;
    mapY = ((mapRows:-1:1)-1)*mapRes;

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
    else
        xlim = [min(mapX(:)),max(mapX(:))];
        ylim = [min(mapY(:)),max(mapY(:))];
    end

    I = mapX>=xlim(1) & mapX<=xlim(2);
    J = mapY>=ylim(1) & mapY<=ylim(2);

    surf(mapX(I),mapY(J),mapZ(J,I),'edgecolor','none');
    axis equal;
    axis_lim = axis;
    axis([axis_lim(1:5),axis_lim(6)*1.5]);
    set(gca,'cameraviewanglemode','manual');

    %axis([xlim,ylim,min(mapZ(:)),max(mapZ(:))*1.5]);

    hold all;

    [Xs,Ys,Zs] = sphere(10);
    h = cell(size(objects));
    Nt = length(t);
    for i=1:frameStep:Nt

        for k=1:length(objects)
            s = objects(k).x(i,1:3);
            X = s(1) + data.r(k)*Xs;
            Y = s(2) + data.r(k)*Ys;
            Z = s(3) + data.r(k)*Zs;
            h{k} = plot3(X,Y,Z,'g-');
        end

        if bMakeGif
            if i==1
                writegif(fig,gifname,dt,false)
            else
                writegif(fig,gifname,dt,true)
            end
        end
        if ~(i==Nt)
            for j=1:viewsmoothing
                view(viewang);
                viewang = viewang + viewrate/viewsmoothing;
                pause(dt*timeScale/viewsmoothing);
            end

            for k=1:length(h)
                delete(h{k});
            end
        end

    end
end