PLANES = ("axial", "coronal", "sagittal")

CLASS_NAMES = ("background", "maxilla", "mandible")

NUM_CLASSES = len(CLASS_NAMES)

_S3_BASE = "https://cvml-datasets.s3.eu-west-3.amazonaws.com/jaws-segmentation/v1/public/2d"

DATASET_URLS = {
    plane: {
        "train": f"{_S3_BASE}/{plane}/train.zip",
        "test": f"{_S3_BASE}/{plane}/test.zip",
    }
    for plane in PLANES
}
