"""
model_student.py — Student Model Architectures for Knowledge Distillation.

Provides simplified model variants with configurable block counts
for the Pareto curve analysis:

    Teacher:      23 blocks (RRDB+Swin+CBAM)  ~16M params
    Student-16:   16 blocks (RRDB only, KD)    ~11M params
    Student-8:    8 blocks  (RRDB only, KD)    ~6M params
    Student-8-noKD: 8 blocks (no KD, baseline)  ~6M params

All students include their own:
    - MC-Dropout layers (for epistemic uncertainty)
    - Uncertainty head (trained with NLL, NOT distilled from teacher)
    - DegradationEstimator + DomainRouter (shared architecture)
"""

from model import FullModel, create_student_model


# =============================================================================
# Student Model Configurations
# =============================================================================

STUDENT_CONFIGS = {
    'student_16': {
        'num_blocks': 16,
        'num_feat': 64,
        'num_grow_ch': 32,
        'description': '16-block RRDB student (KD-trained)',
    },
    'student_8': {
        'num_blocks': 8,
        'num_feat': 64,
        'num_grow_ch': 32,
        'description': '8-block RRDB student (KD-trained)',
    },
    'student_8_nokd': {
        'num_blocks': 8,
        'num_feat': 64,
        'num_grow_ch': 32,
        'description': '8-block RRDB student (NO KD, baseline for Pareto comparison)',
    },
    'student_4_lite': {
        'num_blocks': 4,
        'num_feat': 48,
        'num_grow_ch': 24,
        'description': '4-block lightweight student (max speed)',
    },
}


def create_student(config_name: str = 'student_8', upscale_factor: int = 2, **kwargs) -> FullModel:
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
        upscale_factor=upscale_factor,
        use_log_domain=config.get('use_log_domain', True),
    )
    
    return model


def list_student_configs():
    """Print all available student configurations."""
    print("\nAvailable Student Models:")
    print("-" * 60)
    for name, cfg in STUDENT_CONFIGS.items():
        print(f"  {name:20s} | blocks={cfg['num_blocks']:2d} | "
              f"feat={cfg['num_feat']:2d} | {cfg['description']}")
    print()


if __name__ == '__main__':
    from utils import count_parameters, format_params
    
    list_student_configs()
    
    for name in STUDENT_CONFIGS:
        model = create_student(name)
        print(f"{name:20s} -> {format_params(count_parameters(model))} parameters")
