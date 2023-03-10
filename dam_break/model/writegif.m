function [] = writegif(fig,fname,dt,bAppend)

    frame = getframe(fig);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);

    if ~bAppend 
        imwrite(imind,cm,fname,'gif', 'Loopcount',inf,'DelayTime',dt); 
    else 
        imwrite(imind,cm,fname,'gif','WriteMode','append','DelayTime',dt); 
    end 

end