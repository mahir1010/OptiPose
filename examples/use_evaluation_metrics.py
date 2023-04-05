# Requirements: Both input files must be of the same dimensions (2D/3D).
from cvkit.pose_estimation.data_readers import FlattenedDataStore
from cvkit.pose_estimation.metrics.mpjpe import build_mpjpe_metric
from cvkit.pose_estimation.metrics.pck import build_pck_metric

from cvkit.pose_estimation.config import PoseEstimationConfig
from OptiPose.model.utils import evaluate_predictions

DF3D_config = PoseEstimationConfig('./example_configs/DF3D.yml')

pck_10 = build_pck_metric(DF3D_config.num_parts, 0.10)
pck_10_per_kp = build_pck_metric(DF3D_config.num_parts, 0.10, per_kp=True)

# Edit the following paths. DF3D is ground-truth, sba with 0,2,3,4, and 6 camera indices is baseline, and _model is model output.

baseline = FlattenedDataStore(DF3D_config.body_parts, './sba02346_fly2_004.csv')
pred = FlattenedDataStore(DF3D_config.body_parts, './sba02346_fly2_004_model.csv')
gt = FlattenedDataStore(DF3D_config.body_parts, './df3d_fly2_004.csv')

# baseline = FlattenedDataStore(DF3D_config.body_parts, './sba02346_fly2_005.csv')
# pred = FlattenedDataStore(DF3D_config.body_parts, './sba02346_fly2_005_model.csv')
# gt = FlattenedDataStore(DF3D_config.body_parts, './df3d_fly2_005.csv')

baseline = evaluate_predictions(baseline, gt, [pck_10_per_kp, pck_10], sequence_len=120, batch_size=120)
result = evaluate_predictions(pred, gt, [pck_10_per_kp, pck_10], sequence_len=120, batch_size=120)

print(baseline)
print(result)
