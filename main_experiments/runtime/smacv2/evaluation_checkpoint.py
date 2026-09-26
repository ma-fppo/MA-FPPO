"""Restore released policy weights for evaluation, without training state."""
import torch


def restore_evaluation_policy(backend, path):
    payload = torch.load(path, map_location=backend.device)
    if payload.get('format') != 'macflow_ppo_v1':
        raise ValueError('Expected a discrete MA-FPPO online checkpoint')
    # Check policy semantics; filesystem paths and optimizer settings can relocate.
    for key in ('variant', 'flow_steps', 'temperature', 'torch_history',
                'train_encoder', 'student_latent_std'):
        default = False if key == 'train_encoder' else 0.0 if key == 'student_latent_std' else None
        if payload['config'].get(key, default) != backend.cfg.get(key, default):
            raise ValueError('Evaluation configuration mismatch: ' + key)
    for key in ('model', 'actor', 'reference', 'critic'):
        getattr(backend, key).load_state_dict(payload[key], strict=True)
    if payload.get('mechanism_format'):
        backend.reference_encoder.load_state_dict(payload['reference_encoder'], strict=True)
    backend.reset()
