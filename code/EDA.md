# MedMNIST EDA (data-free quantization study)

All datasets at 64x64 resolution (MedMNIST+ variant).

## Summary

| dataset | task | image (HxWxC) | feature dim | classes | total | train/val/test | imbalance |
|---|---|---|---|---|---|---|---|
| dermamnist | multi-class | 64x64x3 | 12288 | 7 | 10015 | 7007/1003/2005 | 58.66:1 |
| pneumoniamnist | binary-class | 64x64x1 | 4096 | 2 | 5856 | 4708/524/624 | 2.88:1 |
| bloodmnist | multi-class | 64x64x3 | 12288 | 8 | 17092 | 11959/1712/3421 | 2.74:1 |
| pathmnist | multi-class | 64x64x3 | 12288 | 9 | 107180 | 89996/10004/7180 | 1.63:1 |

## Pixel intensity (normalized 0-1)

| dataset | mean | std | min | max |
|---|---|---|---|---|
| dermamnist | 0.6219 | 0.1872 | 0.0 | 1.0 |
| pneumoniamnist | 0.5708 | 0.1735 | 0.0 | 1.0 |
| bloodmnist | 0.7171 | 0.211 | 0.0314 | 1.0 |
| pathmnist | 0.6593 | 0.1911 | 0.0 | 1.0 |

## dermamnist class distribution (train)

| class | count |
|---|---|
| actinic keratoses and intraepithelial carcinoma | 228 |
| basal cell carcinoma | 359 |
| benign keratosis-like lesions | 769 |
| dermatofibroma | 80 |
| melanoma | 779 |
| melanocytic nevi | 4693 |
| vascular lesions | 99 |

## pneumoniamnist class distribution (train)

| class | count |
|---|---|
| normal | 1214 |
| pneumonia | 3494 |

## bloodmnist class distribution (train)

| class | count |
|---|---|
| basophil | 852 |
| eosinophil | 2181 |
| erythroblast | 1085 |
| immature granulocytes(myelocytes, metamyelocytes and promyelocytes) | 2026 |
| lymphocyte | 849 |
| monocyte | 993 |
| neutrophil | 2330 |
| platelet | 1643 |

## pathmnist class distribution (train)

| class | count |
|---|---|
| adipose | 9366 |
| background | 9509 |
| debris | 10360 |
| lymphocytes | 10401 |
| mucus | 8006 |
| smooth muscle | 12182 |
| normal colon mucosa | 7886 |
| cancer-associated stroma | 9401 |
| colorectal adenocarcinoma epithelium | 12885 |
