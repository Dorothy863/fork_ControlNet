from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './exps/models/control_sd21_ini.ckpt'
batch_size = 1
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], accumulate_grad_batches=4)  # But this will be 4x slower

# Train!
trainer.fit(model, dataloader)

"""
多亏了我们组织好的数据集 pytorch 对象和 pytorch_lightning 的强大功能，整个代码非常短。

现在，你可以看看 Pytorch Lightning 官方 DOC，了解如何启用许多有用的功能，如梯度累积、多 GPU 训练、加速数据集加载、灵活的检查点保存等。所有这些只需要大约一行代码。伟大！

请注意，如果您发现 OOM，则可能需要启用 Low VRAM 模式，并且可能还需要使用较小的批量大小和梯度累积。或者你可能还想使用一些 “高级” 技巧，比如 sliced attention 或 xformers。例如：

# Configs
batch_size = 1

# Misc
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], accumulate_grad_batches=4)  # But this will be 4x slower
"""