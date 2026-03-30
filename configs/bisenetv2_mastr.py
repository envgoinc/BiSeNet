
cfg = dict(
    model_type='bisenetv2',
    n_cats=3,
    num_aux_heads=0,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    checkpoint_interval = 1000,  # save every X iterations (pick what you want)
    val_interval = 100,          # run validation every N training iters
    val_num_batches = 0,         # 0 = full val loader; >0 = limit batches per val run

    max_iter=100000,
    dataset='MASTR1325',
    im_root='./datasets/mastr',
    train_im_anns='./datasets/mastr/train.txt',
    val_im_anns='./datasets/mastr/val.txt',
    scales=[0.75, 2.],
    cropsize=[512, 512],
    eval_crop=[512, 512],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=8,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=False,
    

    #wandb stuff
    wandb=True,
    wandb_project="bisenetv2-mastr",
    wandb_entity=None,          
    wandb_run_name="test_segmented_front overfitting",   
    wandb_notes = "testing an overfitting condition version2 ",
     
    wandb_tags=["amp", "ddp"], #amp means automatic mixed precision is enabled in pytorch
)
