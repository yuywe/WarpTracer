"""Headless visualization: python examples/sensors.py [output.png]."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from racesense3d import camera, lidar, CAMERA_TO_BODY, demo_scene


def demo(output=None):
    world=demo_scene()
    position=np.array([[0,0,1]],dtype=np.float32)
    optical=camera()
    laser=lidar(elevation=np.deg2rad(np.linspace(-20,20,16)))
    depth,valid=world.sensor(optical).scan(position,CAMERA_TO_BODY).numpy()
    points=world.sensor(laser).scan(position).points(laser,position).reshape(-1,3)
    points=points[np.isfinite(points).all(axis=1)]
    # Crop the display only; sensor returns above remain unmodified.
    points=points[(points[:,0]>=-5)&(points[:,0]<=9)&(abs(points[:,1])<=10)&(points[:,2]>=-1e-5)&(points[:,2]<=5)]
    fig=plt.figure(figsize=(13,5),layout='constrained')
    ax=fig.add_subplot(121)
    im=ax.imshow(np.ma.array(depth[0],mask=~valid[0]),cmap='viridis')
    ax.set(title='Pinhole depth · 320 × 240',xlabel='Pixel column',ylabel='Pixel row')
    fig.colorbar(im,ax=ax,label='Optical depth (m)',shrink=.8)
    ax=fig.add_subplot(122,projection='3d')
    ax.scatter(*points.T,c=points[:,2],s=1,cmap='viridis')
    ax.scatter([0],[0],[1],c='red',s=30,label='Sensor')
    ax.set(title='16-ring LiDAR · wall, box, ramp, floor',xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
    ax.set_xlim(-5,9);ax.set_ylim(-10,10);ax.set_zlim(0,5)
    ax.legend()
    if output:fig.savefig(output,dpi=160)
    return fig

if __name__=='__main__':
    demo(sys.argv[1] if len(sys.argv)>1 else 'sensor_preview.png')
