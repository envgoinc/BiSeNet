
cfg = dict(
    model_type='bisenetv2',
    n_cats=3,
    num_aux_heads=0,
    lr_start=1e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    checkpoint_interval = 500,  # save every X iterations (pick what you want)
    val_interval = 50,          # run validation every N training iters
    val_num_batches = 0,         # 0 = full val loader; >0 = limit batches per val run

    max_iter=10000,
    dataset='fisheye_small',
    im_root='./datasets/fisheye_small',
    train_im_anns='./datasets/fisheye_small/train.txt',
    val_im_anns='./datasets/fisheye_small/val.txt',
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
    wandb_project="bisenetv2-Fisheye",
    wandb_entity=None,          
    wandb_run_name="Fisheye",   
    wandb_notes = "VERY limited data, should be improved later; this is just to throw points so as long as its just barely correct this should work",
     
    wandb_tags=["amp", "ddp"], #amp means automatic mixed precision is enabled in pytorch
)
