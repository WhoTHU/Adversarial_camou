import os
import numpy as np
import matplotlib.pyplot as plt

prefix = 'results/'
suffix = '/599test_results_tps_iou01_person'

names = [
    'rcnn_sr07',
    'deformable_detr_07',
    'yolov3_sr07',
]

others = [
]

names = [n + suffix for n in names] + others

conf_threshold = 0.5
fig, ax = plt.subplots(2, 2, figsize=(17, 10))
leg = []
for name in names:
    save_dir = os.path.join(prefix + name + '.npz')
    if not os.path.exists(save_dir):
        print('Didn\'t find %s' % save_dir)
    else:
        thetas, info = np.load(save_dir, allow_pickle=True).values()
        confs_part = info[3] #[precision, recall, avg, confs_part]
        leg.append('%s ASR %.4f' % (name.split('/')[0], (confs_part<conf_threshold).mean()))
        ax[0, 0].plot(thetas, confs_part.mean(1))
        ax[0, 0].set_ylim(-0.05, 1.05)
        ax[0, 0].set_xlabel('viewing angles')
        ax[0, 0].set_ylabel('confidence')
        ax[0, 0].set_title('conf mean')
        ax[1, 0].plot(thetas, (confs_part < conf_threshold).mean(1))
        ax[1, 0].set_ylim(-0.05, 1.05) 
        ax[1, 0].set_xlabel('viewing angles')
        ax[1, 0].set_ylabel('success rate')
        ax[1, 0].set_title('detect threshold %.1f' % conf_threshold)
        ax[0, 1].scatter(np.tile(thetas[:, None], (1, confs_part.shape[1])), confs_part.flatten(), s=0.2)
        ax[0, 1].set_ylim(-0.05, 1.05)   
        ax[0, 1].set_title('conf scatter')
        ax[1, 1].plot(thetas, confs_part.max(1))
        ax[1, 1].set_ylim(-0.05, 1.05)    
        ax[1, 1].set_title('max conf')
ax[1, 0].legend(leg)
plt.show()
