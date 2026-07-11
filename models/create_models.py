from .dual_branch import create_model as dual_create_model

def create_model(model_name, num_classes, in_channels, dem_in_channels, encoder_name, encoder_weights):
    """
    统一的模型创建入口。
    所有消融配置都通过 dual_branch.py 中的 create_model 实现。
    忽略 model_name, in_channels, dem_in_channels，由 dual_branch 内部决定。
    """
    # 调用 dual_branch 中的 create_model，只传递必要参数
    return dual_create_model(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights
    )