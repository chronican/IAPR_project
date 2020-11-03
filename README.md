# EE-451-IAPR
Miniproject for EE-451 IAPR 2020 Spring, EPFL.



last update: 26/05/2020

`main.py` is the main function, `utilities.py` `videoIO.py` `segmentation.py`  `classification.py`  are self-evident from there names.

Workflow:

static (done only once): input video -> a stack of input images -> segmentation based on the first image

dynamic (done in a loop): trajectory detection -> detect elements passed-above -> extract formula -> annotate image

final stage (done only once): a stack of output images -> output video