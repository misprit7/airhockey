"""Mixed-height paddle markers must recover the centre, not their centroid."""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vision/bin'))
from camera import backproject_pixels
from mallet_stream import MalletTracker
from track_mallet import MALLET_Z_MM, ARM_Z_MM


@pytest.mark.parametrize('xy', [(1250.,150.), (1500.,500.), (1880.,850.)])
@pytest.mark.parametrize('angle', [0., 1.3, 2.35, 4.])
def test_recovers_centre_at_distinct_marker_heights(xy, angle):
    K=np.array([[800.,0,720.],[0,800.,540.],[0,0,1.]])
    dist=np.zeros(5);rvec=np.array([np.pi,0.,0.]);tvec=np.array([-1000.,500.,1500.])
    def project(points):
        px,_=cv2.projectPoints(np.asarray(points,float),rvec,tvec,K,dist)
        return np.column_stack((px.reshape(-1,2),np.full(len(points),25.)))
    def back(px,z):return backproject_pixels(px,K,dist,rvec,tvec,z)
    tracker=SimpleNamespace(K=K,dist=dist,rvec=rvec,tvec=tvec,_to_table=back,
                            candidates=lambda blobs:(blobs,back(blobs[:,:2],10.)))
    robot=[[xy[0],xy[1],MALLET_Z_MM]]
    for th in (angle,angle+np.pi/2):
        robot.append([xy[0]+26.5*np.cos(th),xy[1]+26.5*np.sin(th),ARM_Z_MM])
    blobs=project(robot)
    got=MalletTracker(tracker).update(blobs)
    assert got is not None
    np.testing.assert_allclose(got[:2],xy,atol=1e-5)
    assert got[2]==3
    assert MalletTracker(tracker).update(blobs[:2]) is None
    # A second equally plausible paddle must be rejected, not arbitrarily chosen.
    other=np.array(robot);other[:,0]-=150
    assert MalletTracker(tracker).update(np.vstack((blobs,project(other)))) is None
