"""Validate every active native SaProt weight; allow only unused HF additions."""
def check_loaded_weights(model, info):
    # The official snapshot omits the contact-regression auxiliary head and an
    # absolute-position table. SaProt uses rotary positions and never calls the
    # contact head when returning residue states. All active parameters are required.
    assert model.config.position_embedding_type == 'rotary'
    allowed={'contact_head.regression.bias','contact_head.regression.weight',
             'embeddings.position_embeddings.weight'}
    missing=info.get('missing_keys',[])
    active=[k for k in missing if k.removeprefix('esm.') not in allowed]
    assert not active, f'Missing active pretrained weights: {active}'
    assert not info.get('mismatched_keys'), info.get('mismatched_keys')
    assert not info.get('error_msgs'), info.get('error_msgs')
    assert model.config.hidden_size==1280 and model.config.num_hidden_layers==33
    return {'unused_missing_hf_parameters':missing,'missing_active_parameters':[]}
