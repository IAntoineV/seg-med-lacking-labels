from typing import Optional
from segmentation_models_pytorch import DeepLabV3Plus

class DeepLabMultiHead(DeepLabV3Plus):
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,
    ):
        super(DeepLabMultiHead, self).__init__(encoder_name=encoder_name, encoder_depth=encoder_depth, encoder_weights=encoder_weights,
                                               encoder_output_stride=encoder_output_stride, decoder_channels=decoder_channels,
                                               decoder_atrous_rates=decoder_atrous_rates,
                                               in_channels=in_channels, classes=classes, activation=activation, upsampling=upsampling,
                                               aux_params=aux_params)
        pass

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks