%% Describes a terrain object derived from a heightmap image.
% This simplifies Y-axis reversal (as images treat Y-down as positive) when
% rendering as well as querying by position.
classdef terrainObject
    
    %% Object properties
    properties (SetAccess = private)
        X % vector of x axis (west-east) distances in meters
        Y % vector of y axis (south-north) distances in meters
        Z % a matrix of height values
        res %map resolution
    end
    
    %% Public methods
    methods
        % Constructor
        % mapZ - raw input heightmap
        % mapRes - map resolution per pixel
        % limits - map extent in meters [xmin,ymin,xmax,ymax]
        function obj = terrainObject(mapZ,mapRes,limits)
            
            obj.X = 0;
            obj.Y = 0;
            obj.Z = 0;
            obj.res = 1;
            if nargin == 0
                return
            end
            if nargin == 1
                obj.res = 1;
            else
                obj.res = mapRes;
            end
            if nargin < 3
                [m,n] = size(mapZ);
                obj.X = ((1:n)-1)' * obj.res;
                obj.Y = ((1:m)-1)' * obj.res;
            else
                obj.X = (limits(1):obj.res:limits(2))';
                obj.Y = (limits(3):obj.res:limits(4))';
            end
            obj.Z = flipud(mapZ);
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
            h = image(obj.X,obj.Y,im);
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