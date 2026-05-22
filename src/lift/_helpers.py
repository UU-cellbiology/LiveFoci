def _make_seg_method(name, params):
    if name == 'cellpose_sam':
        from lift.nuc_segmentation import CP_SAM
        return CP_SAM(
            flow_threshold     = params.get('flow_threshold',      0.0),
            cellprob_threshold = params.get('cellprob_threshold', -0.5),
            scale_factor       = params.get('scale_factor',          1),
        )
    if name == 'cellpose_v3':
        from lift.nuc_segmentation import CP_V3
        return CP_V3(
            flow_threshold     = params.get('flow_threshold',      0.0),
            cellprob_threshold = params.get('cellprob_threshold', -0.5),
            diameter           = params.get('diameter',           140),
        )
    raise ValueError(f"Unknown segmentation method: {name!r}")


def _make_preproc(name):
    if name in (None, 'None', ''):
        return None
    mapping = {
        'wavelet_denoise':   ('lift.registration',     'wavelet_denoise'),
        'threshold':         ('lift.registration',     'threshold'),
        'DOG_filter':        ('lift.registration',     'DOG_filter'),
        'wavelet_filtering': ('lift.nuc_segmentation', 'wavelet_filtering'),
        'contrast_adjuster': ('lift.nuc_segmentation', 'contrast_adjuster'),
    }
    if name not in mapping:
        raise ValueError(f"Unknown preprocessing function: {name!r}")
    module_path, func_name = mapping[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)