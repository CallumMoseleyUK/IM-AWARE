%% Describes a terrain object derived from a heightmap image.
% This simplifies Y-axis reversal (as images treat Y-down as positive) when
% rendering as well as querying by position.
classdef terrainObject
    
    %% Object properties
    properties (SetAccess = private)
        long % vector  of longitude (west-east) breakpoints in degrees
        lat % vector of latitude (south-north) breakpoints in degrees
        X % vector of x axis (west-east) breakpoints in meters
        Y % vector of y axis (south-north) breakpoints in meters
        Z % a matrix of height values
        res %map resolution
        resLat %latitude resolution
        resLong %longitude resolution
    end
    
    %% Public methods
    methods
        % Constructor
        % mapZ - raw input heightmap
        % mapRes - map resolution per pixel
        % limits - map extent in meters [xmin,ymin,xmax,ymax]
%         function obj = terrainObject(mapZ,mapRes,limits)
%             
%             obj.lat = [0,1];
%             obj.X = [0,1];
%             obj.Y = [0,1];
%             obj.Z = zeros(4);
%             obj.res = 1;
%             obj.resX = 1;
%             obj.resY = 1;
%             if nargin == 0
%                 return
%             end
%             [rows,cols] = size(mapZ);
%             if nargin == 1
%                 obj.res = 1;
%             else
%                 obj.res = mapRes;
%                 obj.resY = mapRes;
%                 obj.resX = mapRes;
%             end
%             if nargin < 3
%                 [m,n] = size(mapZ);
%                 obj.X = ((1:n)-1)' * obj.resX;
%                 obj.Y = ((1:m)-1)' * obj.resY;
%                 
%                 obj.lat = 
%             else
%                 obj.X = (limits(1):obj.res:limits(2))';
%                 obj.Y = (limits(3):obj.res:limits(4))';
%             end
%             obj.Z = flipud(mapZ);
%         end
        function obj = terrainObject(mapZ,mapRes,latRange,longRange)
            
            obj.lat = [0,1];
            obj.X = [0,1];
            obj.Y = [0,1];
            obj.Z = zeros(4);
            obj.res = 1;
            obj.resX = 1;
            obj.resY = 1;
            if nargin == 0
                return
            end
            [rows,cols] = size(mapZ);
            if nargin == 1
                obj.res = 1;
            else
                obj.res = mapRes;
                obj.resY = mapRes;
                obj.resX = mapRes;
            end
            [m,n] = size(mapZ);
            obj.lat = linspace(latRange(1),latRange(2),rows);
            obj.long = linspace(longRange(1),longRange(2),cols);
            
            obj.Z = flipud(mapZ);
        end
        
        function pos = coord2Pos(latIn,longIn)
            
            return
        end
        function [latOut,longOut] = pos2Coord(pos)
            
            return
        end
        
        function h = getElevation(obj,pos)
%             [rows,cols] = size(mapZ);
%             locX = s(1)/mapRes + 1; %+1 if indexing starts at 1
%             locY = s(2)/mapRes + 1;
%             pxX = uint32(round(locX));
%             pxY = uint32(round(locY));

            return
        end
        
        function h = getElevationCoords(obj,lat,long)
            
            return
        end
        
        % Returns a submap within the specified limits
        % [xmin,xmax,ymin,ymax] in meters.
        function subObj = submap(obj,limits)
            xmin = limits(1);
            xmax = limits(2);
            ymin = limits(3);
            ymax = limits(4);
            
            I = obj.X>=xmin & obj.X<=xmax;
            J = obj.Y>=ymin & obj.Y<=ymax;
            
            Xs = obj.X(I);
            Ys = obj.Y(J);
            Zs = obj.Z(J,I);
            
            subObj = terrainObject(flipud(Zs),obj.res,[Xs(1),Xs(end),Ys(1),Ys(end)]);
        end
        
        % Render map as a 2D image
        function h = render(obj,fig)
            
            im = repmat(double(obj.Z),1,1,3);
            im = im-min(im(:));
            im = im/max(im(:));
            
            figure(fig);
            h = image(obj.long,obj.lat,im);
            a = gca;
            a.YDir = 'normal';
            axis equal;
        end
    end
    
    %% Private methods
%     methods (Access = private)
%         
%     end

end