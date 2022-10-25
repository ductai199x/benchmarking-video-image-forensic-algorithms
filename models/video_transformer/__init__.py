from .spatial_attn_pretrained_fe import SpatialAttnPretrainedFEPLWrapper

model_config = {
    "img_size": (1080, 1920),
    "in_chans": 3,
    "patch_size": 128,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "qkv_bias": True,
    "mlp_ratio": 4,
    "drop_rate": 0.0,
    "bb1_db_depth": 1,
    "loss_alpha": 0.4,
    "lr": 1e-5,
    "decay_step": 2,
    "decay_rate": 0.85,
    "fe": "mislnet",
    "fe_config": {"patch_size": 128, "num_classes": 33},
    "fe_ckpt": "/media/nas2/deepfake_transformer_models/lab04_220401_mislnet_video_v1_ep=57_vl=0.6216.ckpt",
}
prev_ckpt = "/home/tai/df_models/lab05_220709_spatfe_v11.2.2_mixed_vas_vva_via_ias_iva_iia_ep=09_vl=0.9311.ckpt"

VideoTransformer = SpatialAttnPretrainedFEPLWrapper.load_from_checkpoint(
    prev_ckpt, **model_config
)
