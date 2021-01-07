class YolactEdgePretrained(Model):
    config_class = YolactConfig
    base_model_prefix = 'yolact'

    @classmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike]):
        config = cls.config_class.from_pretrained(name_or_path)
        model = YolactModel(config)
        # model.state_dict(torch.load('yolact.pth'))
        return model

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.BatchNorm2d):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class YolactEdgeModel(YolactEdgePretrained):
    """
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝

    Args:
        config (:class:`YolactConfig`):
        backbone ()
        neck (:class:`Neck`)
        head (:class:`Head`)
        num_classes (:obj:`int`):

    Examples::
        >>> from boda import YolactConfig, YolactModel

        >>> config = YolactConfig()
        >>> model = YolactModel(config)
    """
    model_name = 'yolact'

    def __init__(
        self,
        config,
        backbone = None,
        neck: Neck = None,
        head: Head = None,
        num_classes: int = None,
        **kwargs
    ) -> None:
        ...