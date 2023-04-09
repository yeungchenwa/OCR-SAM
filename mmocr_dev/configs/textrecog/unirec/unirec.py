_base_ = [
    '../_base_/datasets/hiertext.py',
    '../_base_/schedules/schedule_adamw_cos_10e.py',
    '_base_unirec_vit_b.py',
    '../_base_/default_runtime.py',
]

hiertext_textrecog_train = _base_.hiertext_textrecog_train
hiertext_textrecog_train.pipeline = _base_.train_pipeline
hiertext_textrecog_test = _base_.hiertext_textrecog_test
hiertext_textrecog_test.pipeline = _base_.test_pipeline

default_hooks = dict(logger=dict(type='LoggerHook', interval=5))

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=hiertext_textrecog_train)

test_dataloader = dict(
    batch_size=128,
    num_workers=12,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=hiertext_textrecog_test)

val_dataloader = test_dataloader