# Height Bias by HISDAC Land-Use Type

Using the CONUS Test Dataset and reference-BF-conditioned BH inference, we evaluated building-height bias across HISDAC 2015 land-use classes for the CONUS-trained U-Net baseline and single-latent cGAN at epoch 1000.

Across land-use classes and learning rates, U-Net mean BH bias ranged from -3.33 to 2.24 m, whereas single-latent cGAN mean BH bias ranged from -2.67 to -0.12 m. The strongest underprediction occurred for Residential-Owned in the U-Net at learning rate 0.0001, with mean bias of -3.33 m. These results show that the BH underprediction noted in the main validation results is not uniform across land-use contexts, and that land-use classes with taller reference buildings exhibit stronger negative height bias.

Land-use classes ordered by average generated-minus-reference bias:
- Recreational: -2.16 m
- Governmental: -1.92 m
- Residential-Income: -1.73 m
- Vacant Land: -1.70 m
- Residential-Owned: -1.56 m
- Commercial: -1.39 m
- Industrial: -0.76 m
- Agriculture: -0.27 m
