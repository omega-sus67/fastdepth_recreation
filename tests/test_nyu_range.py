import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('acquire_diverse_nyu',Path('scripts/acquire_diverse_nyu.py'))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_parse_range_rejects_truncation_and_unsafe_names():
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w',format=tarfile.USTAR_FORMAT) as archive:
        for name,payload in [('train/room_01/00001.h5',b'abc'),('train/../escape.h5',b'bad'),('train/room_01/00002.h5',b'last')]:
            info=tarfile.TarInfo(name);info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
    data=buffer.getvalue()
    found=module.parse_window(data,512)
    assert [(name,payload) for name,payload,_ in found]==[('train/room_01/00001.h5',b'abc'),('train/room_01/00002.h5',b'last')]
    assert module.parse_window(data[:512],512)==[]
    assert module.parse_window(data[512:],1024)[0][0]=='train/room_01/00002.h5'
    with pytest.raises(ValueError,match='align'):
        module.parse_window(data,1)

def test_difference_hash_is_stable_and_responds_to_horizontal_gradient():
    import numpy as np
    spec=importlib.util.spec_from_file_location('audit_near_duplicates',Path('scripts/audit_near_duplicates.py'))
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    gradient=np.tile(np.linspace(0,255,640,dtype=np.uint8),(3,480,1))
    assert audit.difference_hash(gradient)==(1<<64)-1
    assert audit.difference_hash(gradient[:,:,::-1])==0
    with pytest.raises(ValueError,match='shape'):
        audit.difference_hash(np.zeros((1,1,1),dtype=np.uint8))

def test_location_group_suffix_is_conservative():
    spec=importlib.util.spec_from_file_location('regroup_nyu_scenes',Path('scripts/regroup_nyu_scenes.py'))
    regroup=importlib.util.module_from_spec(spec);spec.loader.exec_module(regroup)
    assert regroup.location_key('kitchen_0045a')==regroup.location_key('kitchen_0045b')=='kitchen_0045'
    assert regroup.location_key('living_room_0019')=='living_room_0019'
    assert regroup.location_key('kitchen_0046a')!='kitchen_0045'
