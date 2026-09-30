import numpy as np
import pytest
import warp as wp
from racesense3d import *

wp.init()
DEVICES = ['cpu'] + (['cuda:0'] if wp.is_cuda_available() else [])

@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def shoot(scene, rays, positions=((0,0,0),), rotations=None, **limits):
    return scene.sensor(rays, len(np.atleast_2d(positions)), **limits).scan(positions, rotations).numpy()


def assert_scan(result, values, valid=True):
    a, b = result
    np.testing.assert_allclose(a, np.broadcast_to(values, a.shape), atol=5e-5, rtol=1e-5)
    np.testing.assert_array_equal(b, np.broadcast_to(valid, b.shape))


def test_oblique_wall(device):
    scene = from_quads([wall_x(5)], device)
    assert_scan(shoot(scene, lidar(np.deg2rad([-45,0,45]))), [np.sqrt(50),5,np.sqrt(50)])


def test_depth_is_axial_and_far_clip_is_axial(device):
    rays = camera(5,3)
    assert 5/rays.factors[0] > 5.5
    assert_scan(shoot(from_quads([wall_x(5)], device), rays,
                      rotations=CAMERA_TO_BODY, far=5.5), 5)


def test_batch_and_point_projection(device):
    scene = from_quads([wall_x(5)], device)
    rays = camera(5,3, fx=3, fy=4, cx=1.8, cy=0.8)
    p = [[0,0,0],[1,0,0]]
    result = scene.sensor(rays, 2).scan(p, CAMERA_TO_BODY)
    assert_scan(result.numpy(), np.array([5,4])[:,None,None])
    points = result.points(rays,p,CAMERA_TO_BODY)
    np.testing.assert_allclose(points[...,0], 5, atol=5e-5)
    local = np.einsum('ij,bhwj->bhwi', CAMERA_TO_BODY.T, points-np.array(p)[:,None,None])
    u = 3*local[...,0]/local[...,2]+1.8
    v = 4*local[...,1]/local[...,2]+0.8
    np.testing.assert_allclose(u, np.broadcast_to(np.arange(5),u.shape), atol=1e-5)
    np.testing.assert_allclose(v, np.broadcast_to(np.arange(3)[None,:,None],v.shape), atol=1e-5)


@pytest.mark.parametrize('distance,near,far,valid', [(2,.02,30,True),(.125,.25,30,False),(5,.02,4,False),(5,.02,5,False),(.25,.25,30,True)])
def test_occlusion_and_limits(device,distance,near,far,valid):
    scene = from_quads([wall_x(10),wall_x(distance)],device)
    assert_scan(shoot(scene,lidar([0]),near=near,far=far), distance if valid else far, valid)


def test_miss_and_two_sided(device):
    scene = from_quads([wall_x(5)[::-1]],device)
    assert_scan(shoot(scene,lidar([0])),5)
    assert_scan(shoot(scene,lidar([np.pi])),30,False)


def test_vertical_lidar_and_camera_orientation(device):
    scene = from_quads([GROUND],device)
    assert_scan(shoot(scene,lidar([0],np.deg2rad([-45,0,45])),[[0,0,1]]),
                [[[np.sqrt(2)],[30],[30]]], [[[True],[False],[False]]])
    assert_scan(shoot(scene,camera(3,3),[[0,0,1]],CAMERA_TO_BODY),
                [[[30]*3,[30]*3,[1.5]*3]], [[[False]*3,[False]*3,[True]*3]])


def test_rigid_transform(device):
    axis=np.array([1,2,3])/np.sqrt(14)
    x,y,z=axis
    K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    a=.61
    R=np.cos(a)*np.eye(3)+np.sin(a)*K+(1-np.cos(a))*np.outer(axis,axis)
    p=np.array([2,-3,1])
    scene=from_quads([np.asarray(wall_x(5))@R.T+p],device)
    assert_scan(shoot(scene,lidar(np.deg2rad([-45,0,45])),[p],R),[np.sqrt(50),5,np.sqrt(50)])


def test_box_against_independent_slab_reference(device):
    lo,hi=np.array([4,-1,-2.]),np.array([6,2,1.])
    rng=np.random.default_rng(71)
    origins=np.vstack([rng.uniform(-3,8,(40,3)),[[5,0,0]]])
    d=rng.normal(size=(100,3))
    d=np.vstack([d,np.eye(3),-np.eye(3)])
    d/=np.linalg.norm(d,axis=1,keepdims=True)
    rays=Rays(d,np.ones(len(d)),(len(d),))
    expected=np.full((len(origins),len(d)),30.)
    valid=np.zeros_like(expected,dtype=bool)
    # Scalar slab intersection, independent of triangle intersection/BVH.
    for b,o in enumerate(origins):
        for r,direction in enumerate(d):
            enter,leave=-np.inf,np.inf
            for k in range(3):
                if abs(direction[k])<1e-12:
                    if not lo[k]<=o[k]<=hi[k]:
                        enter,leave=np.inf,-np.inf
                        break
                else:
                    t1,t2=(lo[k]-o[k])/direction[k],(hi[k]-o[k])/direction[k]
                    enter,leave=max(enter,min(t1,t2)),min(leave,max(t1,t2))
            t=enter if enter>=0 else leave
            if leave>=max(enter,0) and .02<=t<30:
                expected[b,r],valid[b,r]=t,True
    assert_scan(shoot(from_quads(box(lo,hi),device),rays,origins),expected,valid)


def test_resident_pose_buffers_and_reuse(device):
    scene=from_quads([wall_x(5)],device)
    sensor=scene.sensor(lidar([0]))
    result=sensor.scan()
    ptr=result.values.ptr
    assert_scan(result.numpy(),5)
    p=wp.array([[1,0,0]],dtype=wp.vec3,device=device)
    R=wp.array(np.eye(3)[None],dtype=wp.mat33,device=device)
    result2=sensor.scan_device(p,R)
    assert result2.values.ptr==ptr
    assert_scan(result2.numpy(),4)


@pytest.mark.parametrize('call', [lambda:camera(0,3),lambda:camera(fx=0),
    lambda:lidar([]),lambda:lidar([np.nan]),lambda:lidar([0],[2]),
    lambda:Rays([[0,0,0]],[1],(1,)),lambda:box([0,0,0],[0,1,1])])
def test_invalid_sensor_configuration(call):
    with pytest.raises(ValueError):call()


def test_invalid_mesh_pose_and_limits(device):
    with pytest.raises(ValueError):Scene([[0,0,0]]*3,[[0,1,2]],device)
    scene=from_quads([wall_x(5)],device)
    with pytest.raises(ValueError):scene.sensor(lidar([0]),near=5,far=4)
    with pytest.raises(ValueError):scene.sensor(lidar([0])).scan(rotations=np.zeros((3,3)))
    with pytest.raises(ValueError):scene.sensor(lidar([0]),2).scan()


@pytest.mark.skipif(not wp.is_cuda_available(),reason='No CUDA device available')
def test_cpu_cuda_agreement():
    rays=camera(80,60)
    outputs=[]
    for device in ['cpu','cuda:0']:
        outputs.append(shoot(demo_scene(device),rays,[[0,0,1],[1,.2,.8]],CAMERA_TO_BODY))
    np.testing.assert_array_equal(outputs[0][1],outputs[1][1])
    np.testing.assert_allclose(outputs[0][0],outputs[1][0],atol=1e-4,rtol=1e-5)
