from argparse import Namespace
from logging import warn

import timm

from medAI.modeling.vision_transformer import VisionTransformer
from medAI.modeling.swin_transformer import swin_tiny, swin_small, swin_base, swin_large
import torch
from medAI.modeling.ibot.head import DINOHead
from medAI.modeling.ibot.wrappers import MultiCropWrapper
from torch import nn
import torch.distributed as dist
from medAI.utils.distributed import init_distributed, get_rank, get_world_size
import medAI.utils.ibot_dino_utils as utils


MODEL_REGISTRY = {}


def register_model(func):
    MODEL_REGISTRY[func.__name__] = func
    return func


def get_model(name, **kwargs):
    return MODEL_REGISTRY[name](**kwargs)


@register_model
def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


@register_model
def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


@register_model
def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


@register_model
def vit_large(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    return model


@register_model
def vit_small_fft_v2(patch_size=16, **kwargs):
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbedV2, **kwargs)


@register_model
def vit_small_fft_v1(patch_size=16, **kwargs):
    return vit_small(patch_size=patch_size, patch_embed_cls=FFTPatchEmbed, **kwargs)


@register_model
def dinov2_vitb14(**kwargs):
    for k in kwargs.keys():
        warn(f"Unused argument {k}")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    return model


@register_model
def medsam_ibot(**kwargs):
    from medAI.modeling.ibot.medsam_ibot import MedSAMIBot

    kwargs["version"] = "2"
    model = MedSAMIBot(**kwargs)
    model.embed_dim = 768
    return model


def _resnet(name, **kwargs):
    from timm.models import resnet

    model = resnet.__dict__[name](**kwargs)
    embed_dim = model.fc.weight.shape[1]
    model.fc = nn.Identity()

    class ResnetWrapper(nn.Module):
        """Wraps resnet to return last layer feature map as ``patch tokens`` and avgpool as ``class token``"""

        def __init__(self, model: timm.models.ResNet):
            super().__init__()
            self.model = model

        def forward(self, x, mask=None, return_all_tokens=False):
            feature_map = self.model.forward_features(x)
            b, c, h, w = feature_map.shape
            avgpool = self.model.global_pool(feature_map)

            if not return_all_tokens:
                return avgpool

            feature_map_as_tokens = feature_map.reshape(b, c, h * w).permute(0, 2, 1)
            b, c = avgpool.shape
            avgpool = avgpool[:, None, :]

            return torch.cat([avgpool, feature_map_as_tokens], dim=1)

        def get_class_token(self, x) -> torch.Tensor:
            return self(x)

        def get_feature_map(self, x) -> torch.Tensor:
            patch_tokens = self(x, return_all_tokens=True)[:, 1:, :]
            return self._tokens_to_feature_map(patch_tokens)

    model = ResnetWrapper(model)
    model.embed_dim = embed_dim
    return model


@register_model
def resnet18(**kwargs):
    return _resnet("resnet18", **kwargs)


@register_model
def resnet34(**kwargs):
    return _resnet("resnet34", **kwargs)


@register_model
def resnet50(**kwargs):
    return _resnet("resnet50", **kwargs)


@register_model
def ibot_student():
    ...


def get_wrapped_models(conf, checkpoint_path=None):
    student_backbone = get_model(
        conf.model.vit_backbone.arch,
        n_cls_tokens=2,
        masked_im_modeling=conf.model.masked_im_modeling,
        **conf.model.vit_backbone.get("kwargs", {}),
    )
    teacher_backbone = get_model(
        conf.model.vit_backbone.arch,
        n_cls_tokens=2,
        **conf.model.vit_backbone.get("kwargs", {}),
    )
    vit_embed_dim = student_backbone.embed_dim

    student_cnn_backbone = get_model(
        conf.model.cnn_backbone.arch, **conf.model.cnn_backbone.kwargs
    )
    teacher_cnn_backbone = get_model(
        conf.model.cnn_backbone.arch, **conf.model.cnn_backbone.kwargs
    )
    cnn_embed_dim = student_cnn_backbone.embed_dim

    head_kw = dict(
        out_dim=conf.model.n_prototypes,
        act="gelu",
        norm_last_layer=conf.model.norm_last_layer,
    )

    student = MultiCropWrapper(
        student_backbone,
        DINOHead(
            vit_embed_dim,
            **head_kw,
        ),
    )
    teacher = MultiCropWrapper(
        teacher_backbone,
        DINOHead(
            vit_embed_dim,
            **head_kw,
        ),
    )

    cnn_student = MultiCropWrapper(
        student_cnn_backbone,
        DINOHead(
            cnn_embed_dim,
            **head_kw,
        ),
    )

    cnn_teacher = MultiCropWrapper(
        teacher_cnn_backbone,
        DINOHead(
            cnn_embed_dim,
            **head_kw,
        ),
    )

    if checkpoint_path is not None:

        def _select_dict_with_prefix(sd, prefix):
            return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}

        def _replace_prefix_in_dict(sd, old_prefix, new_prefix):
            return {
                k.replace(old_prefix, new_prefix): v
                for k, v in sd.items()
                if k.startswith(old_prefix)
            }

        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        student.load_state_dict(
            _replace_prefix_in_dict(sd["student"], "vit.module.", ""), strict=True
        )
        teacher.load_state_dict(
            _replace_prefix_in_dict(sd["teacher"], "vit.", ""), strict=True
        )
        cnn_student.load_state_dict(
            _replace_prefix_in_dict(sd["student"], "cnn.module.", ""), strict=True
        )
        cnn_teacher.load_state_dict(
            _replace_prefix_in_dict(sd["teacher"], "cnn.module.", ""), strict=True
        )
        print(f"Loaded weights from {checkpoint_path}")

    return dict(
        student=student,
        teacher=teacher,
        cnn_student=cnn_student,
        cnn_teacher=cnn_teacher,
    )


def setup_for_ddp(student, teacher):
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(
            teacher, device_ids=[dist.get_rank()], broadcast_buffers=False
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(
        student, device_ids=[dist.get_rank()], broadcast_buffers=False
    )

    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher, teacher_without_ddp
