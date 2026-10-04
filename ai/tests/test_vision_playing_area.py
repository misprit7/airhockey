"""Rail reflections cannot identify objects; legitimate edge/goal views survive."""
import sys
from pathlib import Path

import numpy as np
import pytest
cv2 = pytest.importorskip('cv2')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vision/bin'))
from camera import backproject_pixels
from playing_area import inside_playing_area, marker_mask, geom
from puck_markers import find_puck, MARK_R
from puck_stream import PuckTracker
from mallet_stream import MalletTracker
from track_mallet import _locate_candidates, MALLET_Z_MM, ARM_Z_MM
from airhockey.vision_service import _detect_loose


K=np.array([[800.,0,720.],[0,800.,540.],[0,0,1.]])
dist=np.zeros(5);rvec=np.array([np.pi,0.,0.]);tvec=np.array([-1000.,500.,1500.])


def project(xy,z,area=30):
    xy=np.asarray(xy,float).reshape(-1,2)
    obj=np.column_stack((xy,np.broadcast_to(z,len(xy))))
    px=cv2.projectPoints(obj,rvec,tvec,K,dist)[0].reshape(-1,2)
    return np.column_stack((px,np.full(len(px),area)))


def corners(xy):
    angle=.3+np.arange(4)*np.pi/2
    return np.asarray(xy)+MARK_R*np.column_stack((np.cos(angle),np.sin(angle)))


def robot(xy):
    angle=2.35+np.arange(2)*np.pi/2
    xy=np.asarray(xy)
    arms=xy+geom.ARM_MARKER_R_MM*np.column_stack((np.cos(angle),np.sin(angle)))
    return np.vstack((project(xy,MALLET_Z_MM),project(arms,ARM_Z_MM)))


def tracker():
    tr=PuckTracker.__new__(PuckTracker)
    tr.K,tr.dist,tr.rvec,tr.tvec=K,dist,rvec,tvec
    tr.glare=None;tr.known_px=np.empty((0,2))
    return tr


def cands(blobs):
    return [(p[2],p[:2]) for p in blobs]


@pytest.mark.parametrize('xy',[(1500,geom.RAIL_MIN_Y+geom.MALLET_RADIUS_MM),
                              (1500,geom.RAIL_MAX_Y-geom.MALLET_RADIUS_MM),
                              (geom.RAIL_MAX_X-geom.MALLET_RADIUS_MM,500)])
def test_elevated_paddle_touching_rail_remains_visible(xy):
    blobs=robot(xy)
    assert marker_mask(blobs[:,:2],K,dist,rvec,tvec).all()
    fix=MalletTracker(tracker()).update(blobs)
    assert fix is not None
    np.testing.assert_allclose(fix[:2],xy,atol=1e-6)
    pose,note=_locate_candidates(cands(blobs),K,dist,rvec,tvec)
    assert pose is not None and note is None
    np.testing.assert_allclose(pose['centre'],xy,atol=1e-6)


@pytest.mark.parametrize('xy',[(1500,geom.RAIL_MIN_Y+10),
                              (1500,geom.RAIL_MAX_Y-10),
                              (geom.RAIL_MAX_X-10,500)])
def test_geometrically_valid_paddle_halfway_through_rail_is_rejected(xy):
    blobs=robot(xy)
    assert MalletTracker(tracker()).update(blobs) is None
    pose,_=_locate_candidates(cands(blobs),K,dist,rvec,tvec)
    # A partial, physically plausible arm cannot become a full pose.
    assert pose is None or pose.get('theta') is None


def test_outside_reflections_do_not_steal_real_robot_or_human():
    actual=[1500.,500.]
    glints=robot([1500.,geom.RAIL_MAX_Y+20])
    blobs=np.vstack((glints,robot(actual)))
    pose,note=_locate_candidates(cands(blobs),K,dist,rvec,tvec)
    assert note is None
    np.testing.assert_allclose(pose['centre'],actual,atol=1e-6)
    # Brightest lone blob is a rail reflection, the dimmer one is the player.
    blobs=np.vstack((project([[700,geom.RAIL_MAX_Y-5]],geom.MALLET_Z_MM,100),
                     project([[500,500]],geom.MALLET_Z_MM,20)))
    puck,player=_detect_loose(cands(blobs),K,dist,rvec,tvec,None)
    assert puck is None
    np.testing.assert_allclose([player['x'],player['y']],[500,500],atol=1e-6)
    fix=MalletTracker(tracker(),markers=1).update(blobs)
    np.testing.assert_allclose(fix[:2],[500,500],atol=1e-6)


@pytest.mark.parametrize('xy',[(800,geom.RAIL_MIN_Y+geom.PUCK_RADIUS_MM),
                              (800,geom.RAIL_MAX_Y-geom.PUCK_RADIUS_MM),
                              (geom.RAIL_MIN_X+geom.PUCK_RADIUS_MM,200)])
def test_puck_against_solid_rail_is_kept(xy):
    found=find_puck(corners(xy))
    assert found is not None
    np.testing.assert_allclose(found[0],xy,atol=1e-6)


def test_invalid_square_cannot_beat_real_puck_or_poison_member_indices():
    impossible=corners([800,geom.RAIL_MAX_Y-10])
    actual=corners([1200,500]);actual[0]+=[.1,-.1]
    assert find_puck(impossible) is None
    found=find_puck(np.vstack((impossible,actual)))
    assert found is not None and set(found[2])=={4,5,6,7}
    np.testing.assert_allclose(found[0],actual.mean(axis=0))


@pytest.mark.parametrize('end',[geom.RAIL_MIN_X-10,geom.RAIL_MAX_X+10])
def test_puck_can_cross_goal_opening_but_not_solid_end_rail(end):
    mouth=(geom.RAIL_MIN_Y+geom.RAIL_MAX_Y)/2
    assert find_puck(corners([end,mouth])) is not None
    assert find_puck(corners([end,200])) is None
    assert not inside_playing_area([[end,mouth]],geom.MALLET_RADIUS_MM)[0]
    blobs=project(corners([end,mouth]),geom.PUCK_MARKER_Z_MM)
    assert marker_mask(blobs[:,:2],K,dist,rvec,tvec).all()


def test_side_reflection_never_becomes_player_even_without_real_player():
    blobs=project([[700,geom.RAIL_MAX_Y-10]],geom.MALLET_Z_MM)
    assert _detect_loose(cands(blobs),K,dist,rvec,tvec,None)==(None,None)
    assert MalletTracker(tracker(),markers=1).update(blobs) is None
