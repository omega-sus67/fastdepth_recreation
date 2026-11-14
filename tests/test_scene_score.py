import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('score_scene_aware',Path('scripts/score_scene_aware.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_scene_score_penalizes_scene_specific_damage():
    assert module.scene_score([0.01,0.09],0.5)==pytest.approx(0.095)
    assert module.scene_score([0.06,0.06],0.5)==pytest.approx(0.09)
    assert module.scene_score([-0.02,-0.01],0.5)==pytest.approx(-0.02)
    with pytest.raises(ValueError):module.scene_score([],0.5)

def test_batched_loss_masks_invalid_depth_and_backpropagates():
    import torch
    spec=importlib.util.spec_from_file_location('train_scene_batch',Path('scripts/train_scene_batch.py'))
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    pred=torch.tensor([[[[2.,4.],[6.,8.]]],[[[1.,2.],[3.,4.]]]],requires_grad=True)
    target=torch.tensor([[[[1.,float('nan')],[0.,4.]]],[[[1.,2.],[3.,2.]]]])
    loss=trainer.masked_image_mae(pred,target)
    torch.testing.assert_close(loss,torch.tensor([2.5,0.5]))
    loss.mean().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad[0,0,0,1]==0 and pred.grad[0,0,1,0]==0
    with pytest.raises(ValueError,match='no valid'):
        trainer.masked_image_mae(pred,torch.zeros_like(pred))
