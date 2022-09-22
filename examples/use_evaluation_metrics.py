# Requirements: Both input files must be of the same dimensions (2D/3D).
from OptiPose.data_store_interface import DannceDataStore
from OptiPose.models.metrics.MPJPE import build_mpjpe_metric
from OptiPose.models.metrics.PCK import build_pck_metric
from OptiPose.models.utils import evaluate_predictions

dannce_bp = ["HeadF", "HeadB", "HeadL", "SpineF", "SpineM", "SpineL", "Offset1", "Offset2", "HipL", "HipR", "ShoulderL",
             "ShoulderR", "KneeR", "KneeL", "ShinL", "ShinR"]

pck_10 = build_pck_metric(len(dannce_bp), 0.1)
mpjpe = build_mpjpe_metric(per_kp=True)

pred = DannceDataStore(dannce_bp, 's1-d1_dannce_pred_save_data_AVG.csv')
gt = DannceDataStore(dannce_bp, 'mocap-s1-d1.csv')

# Inputs: Prediction File, Ground Truth File, list of metrics
# Following will process rows that contains non-nan values in the ground-truth.
# Processing batch size 120x120x16x3 = 120x120 frames at a time
result = evaluate_predictions(pred, gt, [pck_10, mpjpe], sequence_len=120, batch_size=120)
print(result)  # Score 1 PCK@10 Score 2 per keypoint MPJPE
