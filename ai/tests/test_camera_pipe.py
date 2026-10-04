"""A stalled camera pipe must report failure instead of freezing its owner."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vision/bin'))
from camera import Stream


@pytest.mark.parametrize('payload,close,kind',[(b'abc',False,TimeoutError),(b'',True,RuntimeError)])
def test_partial_or_closed_frame_has_bounded_failure(payload,close,kind):
    r,w=os.pipe()
    reader=os.fdopen(r,'rb',buffering=0)
    stream=Stream.__new__(Stream);stream.p=SimpleNamespace(stdout=reader)
    try:
        if payload:os.write(w,payload)
        if close:os.close(w);w=None
        with pytest.raises(kind):stream._read(16,timeout_s=.02)
    finally:
        reader.close()
        if w is not None:os.close(w)


def test_pipe_reader_does_not_consume_following_frame():
    r,w=os.pipe();reader=os.fdopen(r,'rb',buffering=0)
    stream=Stream.__new__(Stream);stream.p=SimpleNamespace(stdout=reader)
    try:
        os.write(w,b'abcdefgh')
        assert stream._read(4)==b'abcd'
        assert stream._read(4)==b'efgh'
    finally:reader.close();os.close(w)
