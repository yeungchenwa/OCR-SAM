dictionary = dict(
    type='Dictionary',
    dict_file=
    '{{ fileDirname }}/../../../dicts/english_digits_symbols_space.txt',
    with_start=True,
    with_end=True,
    same_start_end=True,
    with_padding=True,
    with_unknown=True)

model = dict(
    type='UniRec',
    backbone=dict(
        type='VisionTransformer',
        img_size=(32, 128),
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        pretrained=None),
    decoder=dict(
        type='UniRecDecoder',
        n_layers=6,
        d_embedding=768,
        n_head=8,
        d_model=768,
        d_inner=768 * 4,
        d_k=768 // 8,
        d_v=768 // 8,
        module_loss=dict(
            type='CEModuleLoss', ignore_first_char=True, flatten=True),
        postprocessor=dict(type='AttentionPostprocessor'),
        dictionary=dictionary,
        max_seq_len=48),
    data_preprocessor=dict(
        type='TextRecogDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375]))

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        file_client_args=file_client_args,
        ignore_empty=True,
        min_size=2),
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(type='Resize', scale=(128, 32)),
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(type='STRAugWrapper', op='warp.Curve', mag=3),
                    dict(type='STRAugWrapper', op='warp.Distort', mag=3),
                    dict(type='STRAugWrapper', op='warp.Stretch', mag=3),
                    dict(
                        type='STRAugWrapper', op='geometry.Perspective',
                        mag=3),
                    dict(type='STRAugWrapper', op='geometry.Rotate', mag=3),
                    dict(type='STRAugWrapper', op='geometry.Shrink', mag=3),
                ])
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(type='STRAugWrapper', op='blur.GaussianBlur', mag=2),
                    dict(type='STRAugWrapper', op='blur.DefocusBlur', mag=2),
                    dict(type='STRAugWrapper', op='blur.MotionBlur', mag=2),
                    dict(type='STRAugWrapper', op='blur.GlassBlur', mag=2),
                    dict(type='STRAugWrapper', op='blur.ZoomBlur', mag=2),
                ])
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.25,
        transforms=[
            dict(
                type='RandomChoice',
                transforms=[
                    dict(
                        type='STRAugWrapper', op='noise.GaussianNoise', mag=2),
                ])
        ],
    ),
    dict(
        type='RandomApply',
        prob=0.5,
        transforms=[
            dict(
                type='TorchVisionWrapper',
                op='ColorJitter',
                brightness=0.5,
                saturation=0.5,
                contrast=0.5,
                hue=0.1),
        ]),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(128, 32)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadOCRAnnotations', with_text=True),
    dict(
        type='PackTextRecogInputs',
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'valid_ratio'))
]

tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=0, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=1, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
                dict(
                    type='ConditionApply',
                    true_transforms=[
                        dict(
                            type='ImgAugWrapper',
                            args=[dict(cls='Rot90', k=3, keep_size=False)])
                    ],
                    condition="results['img_shape'][1]<results['img_shape'][0]"
                ),
            ],
            [dict(type='Resize', scale=(128, 32))],
            # add loading annotation after ``Resize`` because ground truth
            # does not need to do resize data transform
            [dict(type='LoadOCRAnnotations', with_text=True)],
            [
                dict(
                    type='PackTextRecogInputs',
                    meta_keys=('img_path', 'ori_shape', 'img_shape',
                               'valid_ratio'))
            ]
        ])
]