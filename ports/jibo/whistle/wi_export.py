"""wi_export.py MODEL.npz OUT.wim: convert train_eval.py --out (logistic regression) to wi_intent's format."""
import struct
import sys

import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
classes = [str(c) for c in z["classes"]]
coef, b = z["coef"].astype(np.float32), z["intercept"].astype(np.float32)
mean, scale = z["mean"].astype(np.float32), z["scale"].astype(np.float32)
with open(sys.argv[2], "wb") as f:
    f.write(b"WIM1" + struct.pack("<ii", len(classes), coef.shape[1]))
    for c in classes:
        e = c.encode()
        f.write(struct.pack("<i", len(e)) + e)
    for a in (mean, scale, coef, b):
        f.write(a.tobytes())
print(len(classes), coef.shape)
