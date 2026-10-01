"""Dim real markers must survive without inventing poses from noise or a puck."""
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'vision/bin'))
import track_mallet as tm
from airhockey.vision_service import VisionService


def render(*, radius=26.5, centre_peak=145, second=False, puck=False):
    K=np.array([[1000.,0,720],[0,1000,540],[0,0,1]])
    dist=np.zeros(5);rvec=np.array([np.pi,0,0]);tvec=np.array([-600.,400,1400])
    field=np.empty((0,2));xy=np.array([1100.,400.]);theta=np.deg2rad(135)
    ys,xs=np.mgrid[:1080,:1440];img=np.full((1080,1440),10.)
    def stamp(point,peak):
        px,_=cv2.projectPoints(np.array([point],float),rvec,tvec,K,dist)
        x,y=px.reshape(2)
        img[:]+=(peak-10)*np.exp(-((xs-x)**2+(ys-y)**2)/8)
    if puck:
        for angle in np.arange(4)*np.pi/2:
            point=xy+tm.geom.PUCK_MARKER_R_MM*np.array([np.cos(angle),np.sin(angle)])
            stamp([*point,tm.geom.PUCK_MARKER_Z_MM],145)
    else:
        for centre in ([xy,xy+[0,200]] if second else [xy]):
            stamp([*centre,tm.MALLET_Z_MM],centre_peak)
            for angle in (theta,theta+np.pi/2):
                point=centre+radius*np.array([np.cos(angle),np.sin(angle)])
                stamp([*point,tm.ARM_Z_MM],240)
    img[:]+=245*np.exp(-((xs-1000)**2+(ys-800)**2)/8)
    return np.clip(img,0,255).astype('uint8'),(K,dist,rvec,tvec,field),xy


def test_dim_centre_recovers_from_same_frame_with_strict_geometry():
    img,args,xy=render()
    candidates=tm.find_candidates(img,tm.field_marker_pixels(*args))
    pose,note=tm._locate_candidates(candidates,*args[:4])
    assert pose is None and 'need 3' in note
    pose,note=tm.locate(img,*args,cands=candidates)
    assert note is None and pose['recovered_dim_marker']
    np.testing.assert_allclose(pose['centre'],xy,atol=.2)
    assert abs(pose['disagree'])<1


@pytest.mark.parametrize('kwargs',[
    {'centre_peak':10}, {'radius':40}, {'second':True}, {'puck':True},
])
def test_recovery_does_not_accept_missing_wrong_or_ambiguous_markers(kwargs):
    img,args,_=render(**kwargs)
    pose,note=tm.locate(img,*args)
    assert pose is None or note is not None


def test_diagnostic_keeps_failure_frame_after_new_frame_arrives(tmp_path):
    service=VisionService()
    service._raw=np.full((20,20),123,dtype=np.uint8)
    service._frame_t=10.;service._pose_t=9.;service._note='missing marker'
    captured=service.diagnostic_snapshot()
    service._raw=np.zeros((20,20),dtype=np.uint8);service._note=None
    service.save_diagnostic(tmp_path,captured)
    assert json.loads((tmp_path/'camera-diagnostic.json').read_text())['note']=='missing marker'
    assert cv2.imread(str(tmp_path/'camera-diagnostic.png'),0).min()==123
