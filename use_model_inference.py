from OptiPose import OptiPoseConfig
from OptiPose.data_store_interface import FlattenedDataStore
from OptiPose.post_processor_interface.PosturalAutoEncoderProcessor import PosturalAutoEncoderProcess

DF3D_config = OptiPoseConfig('./example_configs/DF3D.yml')


pred = FlattenedDataStore(DF3D_config.body_parts, './sba02346_fly2_004.csv')
pred = FlattenedDataStore(DF3D_config.body_parts, './sba02346_fly2_005.csv')

dataset="DF3D"
n_pcm=15
n_cm=5
n_heads=4
output_dims=120
model=PosturalAutoEncoderProcess(DF3D_config,30,n_pcm,n_cm,n_heads,overlap=20,output_dim=output_dims,
                                 weights=f'./OptiPose_weights/{dataset}/optipose-{n_pcm}-{n_cm}-{dataset}-{n_heads}/')
model.PRINT=True
model.process(pred,batch_size=1)
pred.save_file(f'{pred.base_file_path}_model.csv')