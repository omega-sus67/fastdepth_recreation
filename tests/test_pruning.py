import pytest
import torch
from fastdepth_project.models import load_converted
from fastdepth_project.pruning import prune_first_decoder, prune_last_decoder

@pytest.fixture(scope='module')
def base():
    return load_converted('checkpoints/converted/community-skipadd-pruned.pth')[0]

def test_keep_all_is_bit_exact(base):
    candidate,info=prune_last_decoder(base,16)
    x=torch.randn(1,3,32,32)
    with torch.inference_mode():torch.testing.assert_close(candidate(x),base(x),atol=0,rtol=0)
    assert info['removed_channels']==[]

def test_pruned_shapes_and_gradient(base):
    candidate,info=prune_last_decoder(base,12)
    assert candidate.decode_conv5[1][0].out_channels==12
    assert candidate.decode_conv5[1][1].num_features==12
    assert candidate.decode_conv6[0].in_channels==12
    assert len(info['removed_channels'])==4
    y=candidate(torch.randn(1,3,32,32))
    assert y.shape==(1,1,32,32) and torch.isfinite(y).all()
    y.mean().backward()
    assert candidate.decode_conv6[0].weight.grad is not None

def test_rejects_invalid_keep_count(base):
    with pytest.raises(ValueError,match='keep_count'):
        prune_last_decoder(base,17)


def test_first_decoder_keep_all_and_depthwise_groups(base):
    same,_=prune_first_decoder(base,200)
    reduced,info=prune_first_decoder(base,160)
    x=torch.randn(1,3,32,32)
    with torch.inference_mode():
        torch.testing.assert_close(same(x),base(x),atol=0,rtol=0)
        assert reduced(x).shape==(1,1,32,32)
    assert reduced.decode_conv1[1][0].out_channels==160
    assert reduced.decode_conv2[0][0].groups==160
    assert reduced.decode_conv2[1][0].in_channels==160
    assert reduced.decode_conv2[1][0].out_channels==256
    assert len(info['removed_channels'])==40
    reduced(x).mean().backward()
    assert reduced.decode_conv2[1][0].weight.grad is not None

def test_single_channel_ablation_matches_physical_pruning(base):
    candidate,_=prune_first_decoder(base,199,[7])
    def mask(_,inputs,output):
        result=output.clone();result[:,7]=0
        return result
    handle=base.decode_conv2[0].register_forward_hook(mask)
    x=torch.randn(1,3,32,32)
    try:
        with torch.inference_mode():torch.testing.assert_close(base(x),candidate(x),atol=1e-5,rtol=1e-5)
    finally:handle.remove()
    with pytest.raises(ValueError,match='removed_channels'):
        prune_first_decoder(base,199,[7,7])

def test_eight_channel_mask_matches_structural_removal(base):
    removed=[3,12,54,63,68,125,180,191]
    candidate,_=prune_first_decoder(base,192,removed)
    def mask(_,inputs,output):
        result=output.clone();result[:,removed]=0
        return result
    handle=base.decode_conv2[0].register_forward_hook(mask)
    try:
        x=torch.randn(1,3,32,32)
        with torch.inference_mode():
            torch.testing.assert_close(base(x),candidate(x),atol=1e-5,rtol=1e-5)
    finally:handle.remove()
