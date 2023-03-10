%% sim_dambreak - Simulates dam break by modelling released tailings as a group of discrete spherical rolling objects.
%  Callum Moseley 2021   
%
% [t,objects] = sim_dambreak(Nobj,tmax,dt,collision_tick,s0,v0,w0,mapRes,mapZ,pondRadius,vol,rho,c,c_visc,c_col,r_sigma)
%   Input:
%   Nobj - number of discrete objects/particles to simulate
%   tmax - time (seconds) to simulate.
%   dt - simulation time step (seconds).
%   collision_tick - time steps between collision corrections
%   s0 - initial 3D position (meters). (Right handed coordinates)
%   v0 - initial 3D velocity (m/s). (Right handed coordinates)
%   w0 - initial 3D angular velocity (rad/s). (Right handed coordinates)
%   mapRes - map resolution (meters/pixel).
%   mapZ - matrix of map elevation values (meters). Rows increase towards
%   south, columns increase towards east.
%   pondRadius - radius of the tailings pond approximated as circular.
%   vol - volume of released tailings (m^3).
%   rho - density of tailings (kg/m^3)
%   c - linear friction coefficient.
%   c_visc - angular friction coefficient
%   c_col - friction coefficient acting perpendicular to collision normal
%   during a collision
%   r_sigma - standard deviation for normally distributed object radii (set
%   to 0 for no variation)
%
%   Output:
%   t - time vector ranging from 0 seconds to tmax in increments of dt.
%   objects - structure array containing state data over time series t.
%       objects(i).x - state time series of object i.
%       x is a 9 element state vector formatted [position, velocity, angular
%       velocity].
%   data - additional simulation data
%
%
function [t,objects,data] = sim_dambreak_stochastic(Nobj,tmax,dt,collision_tick,...
    s0,v0,w0,mapRes,mapZ,pondRadius,vol,rho,c,c_visc,c_col,r_sigma)
    
    % Maximum number of iterations without collision before a particle is flagged inactive 
    maxCollisionIter = Inf;
    
    % Initial state
    x0 = [s0(:); v0(:); w0(:); 0];
    
    % Time vector
    t = (0:dt:tmax)';
    
    % Instantiate objects to simulate
    fluid_height = vol/(pi*pondRadius^2);
    Nradial = floor( (Nobj * pondRadius / (pi*fluid_height))^(1/3));
    r0 = pondRadius/(2*Nradial);
    Nlayers = round(fluid_height/(2*r0));
    objects = init_objects(t,x0,r0,pondRadius,Nlayers);
    Nobj = length(objects);
    %fprintf('N = %i, h = %1.5f',Nobj,fluid_height);
    
    % Base mass of each sphere (kg)
    m0 = rho*vol/Nobj;
    
    % Vary object sizes by a normal distribution
    sigma = r_sigma*r0;
    r =max(normrnd(r0,sigma,[Nobj,1]), 0.1);
    v_obj0 = (4/3)*pi*r0^2;
    v_obj = (4/3)*pi*r.^2;
    rho0 = m0/v_obj0;
    m = rho0*v_obj;
    
    % Moment of inertia for a sphere (kg m^2)
    I0 = (2/5)*m0.*r0.^2;
    I = (2/5)*m.*r.^2;
    % Linear friction coefficient between each sphere and the terrain
    c = m*c;
    % Rotational damping coefficient
    c_visc = I*c_visc;
    % Collision friction
    % this is implicitly scaled by m in the collision function.
    %c_col = m0*c_col;
    
    % Data
    data.Nobj = Nobj;
    data.m = m;
    data.I = I;
    data.r = r;
    data.c = c;
    data.c_visc = c_visc;
    data.c_col = c_col;

    % Integrate system with respect to time
    X = zeros(length(objects),length(x0));
    collision_count = 0;
    for i=2:length(t)
        collide_ledger = true(size(objects));
        for j=1:length(objects)
            % --Kinematic equations of motion
            xi = objects(j).x(i-1,:);
            
            % Skip inactive particles
            if xi(10)>maxCollisionIter
                objects(j).x(i,:) = xi;
                continue;
            end
            
            [dx, objects(j).surfdata] = eom_ball(objects(j).surfdata,t(i),xi',mapRes,mapZ,m(j),I(j),c(j),c_visc(j),r(j));
            xi = xi + dx' * dt;
            
            si = xi(1:3);
            vi = xi(4:6);
            wi = xi(7:9);
            
            collide_ledger(j) = sum(dx(1:6).^2) < 0.1;

            % -- Collision with map surface
            % constrain z position to above or at ground level
            Zmin = objects(j).surfdata.Z+r(j);
            if si(3)<Zmin
                si(3) = Zmin;
                n = objects(j).surfdata.n;
                dV = vi*n;
                vi = vi - min(dV,0) * n';
            end
            
            % -- Update object state
            xi = [si,vi,wi,xi(10)];
            objects(j).x(i,:) = xi;
            
            % -- Collect matrix of all current object states (one row per object)
            % so that collision can be calculated without passing the
            % entire time-series.
            X(j,:) = xi;
            
        end
        if collision_count==0
            for j=1:length(objects)
                if ~collide_ledger(j)
                    continue;
                end
                % -- Calculate collision with other objects
                % Modelled as rigid-bodies, this is an instantanteous impulse
                % rather than a PDE, and is not affected by time step length.
                % Note that this must happen after integrating the PDEs so that
                % most up-to-date state is available for all objects for this
                % step in time.
                [X] = collision(j,X,m,r,c_col);
            end
        end
        collision_count = collision_count+1;
        if collision_count==collision_tick
            collision_count = 0;
        end
        for k=1:size(X,1)
            objects(k).x(i,:) = X(k,:);
        end
        if mod(i,100)==0
            fprintf('t = %1.1f\n',t(i));
        end
    end
    
    % Reduce time step to once every collision tick (add as a separate
    % input maybe, DT in addition to dt)
    I = 1:collision_tick:length(t);
    t = t(I);
    for i=1:length(objects)
        objects(i).x = objects(i).x(I,:);
    end
end

% Initialises spherical objects and places them in a cylindrical
% distribution within the pond
% Object states are:
% -position (3),
% -velocity (3),
% -angular velocity (3),
function [objects] = init_objects(t,x0,r,pondRadius,Nlayers)
    % Place objects in concentric rings within the pond
    Nring = max(floor(pondRadius/(2*r)),1);
    k = 0;
    for l=1:Nlayers
        for n=1:Nring
            %dT = 2/(2*n-1); %angle increment for parametric circle equation
            Ncirc = round(2*pi*n);
            %dT = 2*pi/Ncirc;
            a = (2*n-1)*r; %object distance from pond centre
            T = linspace(0,2*pi,Ncirc);
            for j=1:length(T)
                k = k+1;
                objects(k).x = zeros(length(t),length(x0));
                objects(k).x(1,:) = x0;

                pos = [a*cos(T(j)), a*sin(T(j)), 2*r*(l-1)];
                objects(k).x(1,1:3) = objects(k).x(1,1:3) + pos;
                
                % Assign null surface data
                objects(k).surfdata.pxX = -1;
                objects(k).surfdata.pxY = -1;
                
                
            end
        end
    end
end

% Adjusts all object states instantaneously as an impulse to simulate inelastic
% rigid-body collision for object j.
function [X] = collision(j,X,m,r,c_fric)
    
    minDist2 = Inf;
    bCollisionDetected = false;
    
    for k=1:size(X,1)
        % Do not calculate collision with self or objects not in contact
        if k==j
            continue;
        end
        
        % Squared distance threshold for collision to occur
        distThresh = r(k)+r(j);
        distThresh2 = distThresh^2;

        xa = X(k,:); %state vector a
        xb = X(j,:); %state vector b
        disp = xb(1:3) - xa(1:3); %vector displacement between objects
        dist2 = disp*disp'; % squared distance between objects
        
        %minDist2 = min(minDist2,dist2); %calculate closest particle
        
        if dist2>=distThresh2 || dist2==0 %this check is faster than unsquared distance by skipping calculation of square roots
            continue; %maybe pass xa and xb to second function here?
        end
        bCollisionDetected = true;
        
        % Calculate collision normal
        dist = sqrt(dist2);
        n = disp/dist;
        
        % Calculate relative velocity
        Vba = xb(4:6) - xa(4:6);
        
        % Calculate specific impulse (assumes inelastic
        % collision, plus friction)
        %dv = 0.5*(ua-ub)*n;
        mRatio_b = m(k)/(m(k)+m(j));
        mRatio_a = m(j)/(m(k)+m(j));
        dv_b = (c_fric-mRatio_b)*(Vba*n')*n - Vba*c_fric;
        dv_a = (c_fric-mRatio_a)*(Vba*n')*n - Vba*c_fric;
        
        % Adjust velocity state manually
        xa(4:6) = xa(4:6) - dv_a;
        xb(4:6) = xb(4:6) + dv_b;
        
        % Adjust position to correct encroachment of objects on each
        % other's bounding sphere.
        encroachDist = distThresh - dist;
        xa(1:3) = xa(1:3) - mRatio_a*encroachDist*n;
        xb(1:3) = xb(1:3) + mRatio_b*encroachDist*n;
        
        X(k,:) = xa;
        X(j,:) = xb;
    end
    
    % Flag particle for removal if it moves too far from others
    %bRemove = minDist2>(100*distThresh);
    %X(j,10) = ~bRemove;
    if ~bCollisionDetected
        X(j,10) = X(j,10) + 1;
    else
        X(j,10) = 0;
    end
end

% eom_ball - Describes the ODEs of motion of a ball in freefall on terrain.
%   t - time
%   m - mass
%   I - moment of inertia (scalar, as the ball is assumed to be uniformly
%     dense and spherical)
%   c - damping
%   r - ball radius
%   mapZ - height map in meters
%   mapRes - resolution of map in meters
%
%   dx - time derivative of the state vector.
%   Z - height of terrain at current position.
%   n - normal vector of terrain at current position.
%
%   Axis system (right handed)
%   - Positive x - east
%   - Positive y - north
%   - Positive z - upwards
%
% Optimisation note: Matlab will copy mapZ into function workspace if
% modified inside eom_ball. This is a very big matrix, so only reference
% it, never modify it here.
% Also, only pass surface data, not the whole object, as the object
% contains large time-series data which will be copied if the object is
% modified.
function [dx, surfdata] = eom_ball(surfdata,t,x,mapRes,mapZ,m,I,c,c_visc,r)
    g = 9.81;
    
    s = x(1:3); %[x,y,z] displacement
    ds = x(4:6); %[x,y,z] velocity
    w = x(7:9); %[x,y,z] angular velocity
    
    % Calculate pixel position on map
    [rows,cols] = size(mapZ);
    locX = s(1)/mapRes + 1; %+1 if indexing starts at 1
    locY = rows - s(2)/mapRes;
    pxX = uint32(round(locX));
    pxY = uint32(round(locY));
    
    % Freeze motion if map edge is reached
    if pxX<=1 || pxX>=cols | pxY<=1 || pxY>=rows
        dx = x*0;
        surfdata.Z = 0;
        surfdata.n = [0;0;1];
        return;
    end
    
    % If the object has moved to a new pixel/cell in the map, update surface norm
    % and adjacent height values.
    if pxX==surfdata.pxX && pxY==surfdata.pxY
        
    else
        % Update current pixel/cell
        surfdata.pxX = pxX;
        surfdata.pxY = pxY;
        
        % Get height value from adjacent pixels
        surfdata.h_east = mapZ(pxY,pxX+1);
        surfdata.h_west = mapZ(pxY,pxX-1);
        surfdata.h_north = mapZ(pxY-1,pxX);
        surfdata.h_south =  mapZ(pxY+1,pxX);

        % Calculate surface normal (n) and tangent (u)
        % - convert height values to double *before* subtraction, as
        % they are unsigned and therefore negative gradients are ignored.
        dh_dx = (double(surfdata.h_east) - double(surfdata.h_west))...
            ./ (2*mapRes);
        dh_dy = (double(surfdata.h_north) - double(surfdata.h_south))...
            ./ (2*mapRes);
        dh_dz = -1;

        surfdata.n = -[dh_dx; dh_dy; dh_dz] ./ sqrt(dh_dx^2 + dh_dy^2 + 1);
    end
    % Current map Z minimum (for elevation constraint)
    %Z = min(min(double(mapZ(pxY+[-1,1],pxX+[-1,1]))));
    ax = (mod(locX,1) + 1)/2;
    ay = (mod(locY,1) + 1)/2;
    surfdata.Z = (double(surfdata.h_south) + ay*(double(surfdata.h_north)-double(surfdata.h_south))...
        + double(surfdata.h_west) + ax*(double(surfdata.h_east)-double(surfdata.h_west)))/2;

    % Calculate forces
    W = [0;0;-m*g]; %weight
    f = -c*(ds + cross(r*surfdata.n,w)); %friction
    torque = cross(-r*surfdata.n, f) - c_visc*w;
    
    % Calculate ODEs
    % dds = (W - (W'*n)*n + f) / m;
    dds = (W + f) / m; % translational acceleration
    dw = torque / I; % angular acceleration
    
    dx = [ds; dds; dw; 0]; %full state vector time derivative
end