"""Physically prune the final decoder features without touching skip-add paths."""
import copy
import torch
from .models import from_spec, module_spec


def _child(spec, path):
    node=spec
    for name in path.split('.'):
        matches=[child for key,child in node['children'] if key==name]
        if len(matches)!=1:
            raise ValueError(f'Expected exactly one architecture node at {path}')
        node=matches[0]
    return node


def prune_last_decoder(model, keep_count):
    """Keep channels with largest absolute final-output weights; fixed structural rule.

    This only targets decode_conv5 pointwise output, which is after all skip additions.
    It must also slice that layer's batch normalization and the next convolution's input.
    """
    spec=copy.deepcopy(module_spec(model));state=model.state_dict()
    point=_child(spec,'decode_conv5.1.0');bn=_child(spec,'decode_conv5.1.1');final=_child(spec,'decode_conv6.0')
    if point['type']!='Conv2d' or bn['type']!='BatchNorm2d' or final['type']!='Conv2d':
        raise ValueError('Unexpected decoder structure')
    count=point['parameters']['out_channels']
    if final['parameters']['in_channels']!=count or bn['parameters']['num_features']!=count:
        raise ValueError('Coupled channel counts do not match')
    if final['parameters']['groups']!=1 or point['parameters']['groups']!=1:
        raise ValueError('Unsupported grouped pointwise convolution')
    if not 1<=keep_count<=count:raise ValueError('keep_count out of range')
    score=state['decode_conv6.0.weight'].detach().abs().sum(dim=(0,2,3))
    # Stable deterministic tie-breaking by original channel index.
    keep=sorted(sorted(range(count),key=lambda i:(-float(score[i]),i))[:keep_count])
    index=torch.tensor(keep,dtype=torch.long,device=score.device)
    point['parameters']['out_channels']=keep_count
    bn['parameters']['num_features']=keep_count
    final['parameters']['in_channels']=keep_count
    new_state={name:value.detach().clone() for name,value in state.items()}
    for name in ('decode_conv5.1.0.weight','decode_conv5.1.1.weight','decode_conv5.1.1.bias',
                 'decode_conv5.1.1.running_mean','decode_conv5.1.1.running_var'):
        if name not in new_state:raise ValueError(f'Missing required weight {name}')
        new_state[name]=new_state[name].index_select(0,index)
    if 'decode_conv5.1.0.bias' in new_state:
        new_state['decode_conv5.1.0.bias']=new_state['decode_conv5.1.0.bias'].index_select(0,index)
    new_state['decode_conv6.0.weight']=new_state['decode_conv6.0.weight'].index_select(1,index)
    new_model=from_spec(spec)
    new_model.load_state_dict(new_state,strict=True)
    return new_model.eval(),{'original_channels':count,'kept_channels':keep,'removed_channels':sorted(set(range(count))-set(keep)),
                             'criterion':'largest absolute final 1x1 output weight, deterministic index tie-break',
                             'scope':'decode_conv5 pointwise outputs and BN, decode_conv6 input; after skip additions'}


def prune_first_decoder(model, keep_count, removed_channels=None):
    """Prune first decoder outputs and the next depthwise/pointwise inputs.

    Stage 2 output and its subsequent skip addition are preserved. This is one
    deliberately narrow structural pattern, not a general dependency solver.
    """
    spec=copy.deepcopy(module_spec(model));state=model.state_dict()
    point=_child(spec,'decode_conv1.1.0');bn1=_child(spec,'decode_conv1.1.1')
    dw=_child(spec,'decode_conv2.0.0');bn2=_child(spec,'decode_conv2.0.1');next_point=_child(spec,'decode_conv2.1.0')
    count=point['parameters']['out_channels']
    if not (bn1['parameters']['num_features']==count and dw['parameters']['in_channels']==count
            and dw['parameters']['out_channels']==count and dw['parameters']['groups']==count
            and bn2['parameters']['num_features']==count and next_point['parameters']['in_channels']==count
            and point['parameters']['groups']==1 and next_point['parameters']['groups']==1):
        raise ValueError('Unexpected coupled depthwise decoder structure')
    if not 1<=keep_count<=count:raise ValueError('keep_count out of range')
    score=state['decode_conv2.1.0.weight'].detach().abs().sum(dim=(0,2,3))
    if removed_channels is None:
        keep=sorted(sorted(range(count),key=lambda i:(-float(score[i]),i))[:keep_count])
        criterion='largest L1 input-channel weight in second decoder pointwise convolution, deterministic index tie-break'
    else:
        removed=list(removed_channels)
        if len(removed)!=count-keep_count or len(set(removed))!=len(removed) or any(not isinstance(i,int) or i<0 or i>=count for i in removed):
            raise ValueError('Invalid removed_channels')
        keep=sorted(set(range(count))-set(removed))
        criterion='explicit removed channels from external selection'
    index=torch.tensor(keep,dtype=torch.long,device=score.device)
    point['parameters']['out_channels']=keep_count;bn1['parameters']['num_features']=keep_count
    for key in ('in_channels','out_channels','groups'):dw['parameters'][key]=keep_count
    bn2['parameters']['num_features']=keep_count;next_point['parameters']['in_channels']=keep_count
    new_state={name:value.detach().clone() for name,value in state.items()}
    for prefix in ('decode_conv1.1.1','decode_conv2.0.1'):
        for suffix in ('weight','bias','running_mean','running_var'):
            name=f'{prefix}.{suffix}'
            if name not in new_state:raise ValueError(f'Missing required weight {name}')
            new_state[name]=new_state[name].index_select(0,index)
    for name in ('decode_conv1.1.0.weight','decode_conv2.0.0.weight'):
        new_state[name]=new_state[name].index_select(0,index)
    for name in ('decode_conv1.1.0.bias','decode_conv2.0.0.bias'):
        if name in new_state:new_state[name]=new_state[name].index_select(0,index)
    new_state['decode_conv2.1.0.weight']=new_state['decode_conv2.1.0.weight'].index_select(1,index)
    candidate=from_spec(spec);candidate.load_state_dict(new_state,strict=True)
    return candidate.eval(),{'original_channels':count,'kept_channels':keep,
            'removed_channels':sorted(set(range(count))-set(keep)),
            'criterion':criterion,
            'scope':'decoder stage 1 pointwise output/BN and stage 2 depthwise groups/BN and pointwise input; skip output unchanged'}
