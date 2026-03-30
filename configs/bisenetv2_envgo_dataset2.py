cfg = dict(
    model_type='bisenetv2',
    n_cats=3, 
    num_aux_heads=4,
    lr_start=1e-3, #pontentially can be a red herrinng 
    weight_decay=5e-4,
    warmup_iters=1000,
    checkpoint_interval=250,
    val_interval=250,
    val_num_batches=100,
    max_iter=450000,
    dataset="MultiSourceJsonDataset", #very important this matches the correct loader; this loader I made should be good. 
    datasets=[
        {"images_dir": "/app/mar23rd_data/merged/images", "labels_dir": "/app/mar23rd_data/merged/labels_png_coloured", "annpath": "/app/mar23rd_data/merged/merged_splits_cleaned.json"},
        {"images_dir": "/app/BiSeNet/data_birdseye_long_selection/images", "labels_dir": "/app/BiSeNet/data_birdseye_long_selection/labels_coloured", "annpath": "/app/BiSeNet/data_birdseye_long_selection/matched_splits.fixed2.json"},
        # {"images_dir": "/app/birdseye_run_6/frames/rgb", "labels_dir": "/app/birdseye_run_6/frames/mask_coloured", "annpath": "/app/birdseye_run_6/generated_splits.json"},
        # {"images_dir": "/app/birdseye_run_7/frames/rgb", "labels_dir": "/app/birdseye_run_7/frames/mask_coloured", "annpath": "/app/birdseye_run_7/generated_splits.json"},
        # {"images_dir": "/app/birdseye_run_9/frames/output_rgb", "labels_dir": "/app/birdseye_run_9/frames/output_mask_coloured", "annpath": "/app/birdseye_run_9/generated_splits.json"},
        {"images_dir": "/app/birdseye_run_10/images", "labels_dir": "/app/birdseye_run_10/labels_cleaned", "annpath": "/app/birdseye_run_10/generated_splits.json"},
        {"images_dir": "/app/birdseye_run_11/images", "labels_dir": "/app/birdseye_run_11/labels_cleaned", "annpath": "/app/birdseye_run_11/generated_splits.json"},
        {"images_dir": "/app/birdseye_run_12/images", "labels_dir": "/app/birdseye_run_12/labels_cleaned", "annpath": "/app/birdseye_run_12/generated_splits.json"},

    ],
    lb_ignore=255, #this is a uint8 class, not a colour / pixel value that you ignore.                         
    mean=(0.4, 0.4, 0.4),                   
    std=(0.2, 0.2, 0.2),                 
    label_color_map=[                       
        [255, 0,   0],                       #   1 = self  (red)
        [0,   255, 0],                       #   2 = objects (green)
        [0,   0,   255],                     #   3 = water (blue)
    ],
    scales=[0.75, 2.0],
    # cropsize=[896, 1184], # very important!! height then width; an important change; use [512, 512] for training and [896, 1184] for exporting and compiling
    cropsize=[512, 512],
    eval_crop=[512, 512],
    eval_scales=[0.9, 1.0, 1.75],
    ims_per_gpu=8,
    eval_ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=False,
    wandb=True,
    wandb_project="bisenetv2-BEV",
    wandb_entity=None,
    wandb_run_name="new_trimmed + new sim data",
    wandb_notes="augmentations included",
    wandb_tags=["amp", "ddp"],
)