"""
model_student.py — Student Model Architectures for Knowledge Distillation & Edge Deployment.

Provides simplified model variants with configurable block counts and structural
reparameterization (RepBlock) for the Pareto curve analysis:

    Teacher:          23 blocks (RRDB + MDTA + Manhattan Attention)  ~16.9M params
    Student-16-Rep:   16 RepBlocks (Multi-branch train -> 1-conv deploy) ~5.2M params
    Student-8-Rep:    8 RepBlocks  (Multi-branch train -> 1-conv deploy) ~2.8M params
    Student-16:       16 blocks (RRDB + MDTA, KD)                    ~11.5M params
    Student-8:        8 blocks  (RRDB + MDTA, KD)                    ~6.4M params
    Student-4-Lite:   4 blocks  (Max speed edge inference)          ~1.8M params

All students support:
    - Zero-cost inference acceleration via `switch_to_deploy()` (for RepBlock variants)
    - MDTA Global Transposed Attention / Swin Attention
    - Decoupled two-stage SR head
"""

from model import FullModel, create_student_model


# =============================================================================
# Student Model Configurations
# =============================================================================

STUDENT_CONFIGS = {
    'student_16_rep': {
        'num_blocks': 16,
        'num_feat': 64,
        'num_grow_ch': 32,
        'use_repblock': True,
        'attention_type': 'mdta',
        'description': '16-block RepBlock student (Multi-branch train -> 1-conv deploy, max accuracy)',
    },
    'student_8_rep': {
        'num_blocks': 8,
        'num_feat': 64,
        'num_grow_ch': 32,
        'use_repblock': True,
        'attention_type': 'mdta',
        'description': '8-block RepBlock student (Multi-branch train -> 1-conv deploy, 80+ FPS)',
    },
    'student_16': {
        'num_blocks': 16,
        'num_feat': 64,
        'num_grow_ch': 32,
        'use_repblock': False,
        'attention_type': 'mdta',
        'description': '16-block RRDB student (KD-trained)',
    },
    'student_8': {
        'num_blocks': 8,
        'num_feat': 64,
        'num_grow_ch': 32,
        'use_repblock': False,
        'attention_type': 'mdta',
        'description': '8-block RRDB student (KD-trained)',
    },
    'student_8_nokd': {
        'num_blocks': 8,
        'num_feat': 64,
        'num_grow_ch': 32,
        'use_repblock': False,
        'attention_type': 'mdta',
        'description': '8-block RRDB student (NO KD, baseline for Pareto comparison)',
    },
    'student_4_lite': {
        'num_blocks': 4,
        'num_feat': 48,
        'num_grow_ch': 24,
        'use_repblock': True,
        'attention_type': 'mdta',
        'description': '4-block lightweight RepBlock student (ultra-fast embedded inspection)',
    },
}


def create_student(config_name: str = 'student_8_rep', upscale_factor: int = 2, **kwargs) -> FullModel:
    """Create a student model from a named configuration.
    
    Args:
        config_name: Key from STUDENT_CONFIGS.
        upscale_factor: Super-resolution factor (default: 2).
        **kwargs: Override any config parameter.
        
    Returns:
        FullModel instance configured as a student.
    """
    if config_name not in STUDENT_CONFIGS:
        raise ValueError(f"Unknown student config: {config_name}. "
                         f"Available: {list(STUDENT_CONFIGS.keys())}")
    
    config = STUDENT_CONFIGS[config_name].copy()
    config.update(kwargs)
    
    model = create_student_model(
        num_blocks=config['num_blocks'],
        num_feat=config.get('num_feat', 64),
        num_grow_ch=config.get('num_grow_ch', 32),
        attention_type=config.get('attention_type', 'mdta'),
        upscale_factor=upscale_factor,
        use_log_domain=config.get('use_log_domain', True),
        use_repblock=config.get('use_repblock', False),
    )
    
    return model


def list_student_configs():
    """Print all available student configurations."""
    print("\nAvailable Student Models:")
    print("-" * 75)
    for name, cfg in STUDENT_CONFIGS.items():
        rep_str = "RepBlock" if cfg.get('use_repblock') else "RRDB"
        print(f"  {name:18s} | blocks={cfg['num_blocks']:2d} | "
              f"type={rep_str:8s} | {cfg['description']}")
    print()


if __name__ == '__main__':
    from utils import count_parameters, format_params
    
    list_student_configs()
    
    for name in STUDENT_CONFIGS:
        model = create_student(name)
        print(f"{name:20s} -> {format_params(count_parameters(model))} parameters")
