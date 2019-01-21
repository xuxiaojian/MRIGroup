from tensorboardX import SummaryWriter
from util import imgProcess


class TBXWriter(object):

    def __init__(self, path):
        self.writer = SummaryWriter(path)
        self.config_init()

    def minibatch_before_batch(self, loss, psnr, ssim, step):
        self.writer.add_scalar(tag='train_batch/before_loss', scalar_value=loss, global_step=step)
        self.writer.add_scalar(tag='train_batch/before_psnr', scalar_value=psnr, global_step=step)
        self.writer.add_scalar(tag='train_batch/before_ssim', scalar_value=ssim, global_step=step)

    def minibatch_after_batch(self, loss, psnr, ssim, step):
        self.writer.add_scalar(tag='train_batch/after_loss', scalar_value=loss, global_step=step)
        self.writer.add_scalar(tag='train_batch/after_psnr', scalar_value=psnr, global_step=step)
        self.writer.add_scalar(tag='train_batch/after_ssim', scalar_value=ssim, global_step=step)

    def train_epoch(self, loss, psnr, ssim, imgs, step):
        self.writer.add_scalar(tag='train_epoch/average_loss', scalar_value=loss, global_step=step)
        self.writer.add_scalar(tag='train_epoch/average_psnr', scalar_value=psnr.mean(), global_step=step)
        self.writer.add_scalar(tag='train_epoch/average_ssim', scalar_value=ssim.mean(), global_step=step)

        width_img = imgs.shape[1]
        height_img = imgs.shape[2]
        channel = imgs.shape[3]

        for i in range(imgs.shape[0]):
            self.writer.add_scalar(tag='train_epoch/index_%d_psnr' % i, scalar_value=psnr[i], global_step=step)
            self.writer.add_scalar(tag='train_epoch/index_%d_ssim' % i, scalar_value=ssim[i], global_step=step)
            for j in range(channel):
                self.writer.add_image(tag='train_epoch/prediction_index_%d_chaneel_%d' % (i, j),
                                      img_tensor=imgProcess.normalize(imgs[i].reshape([width_img, height_img])),
                                      global_step=step)

    def imgs_train_init(self, x_imgs, y_imgs):
        width_img = x_imgs.shape[1]
        height_img = x_imgs.shape[2]
        channel = x_imgs.shape[3]

        for i in range(x_imgs.shape[0]):
            for j in range(channel):
                self.writer.add_image(tag='train_epoch/input_index_%d_chaneel_%d' % (i, j),
                                      img_tensor=imgProcess.normalize(x_imgs[i].reshape([width_img, height_img])),
                                      global_step=0)
                self.writer.add_image(tag='train_epoch/ground-truth_index_%d_chaneel_%d' % (i, j),
                                      img_tensor=imgProcess.normalize(y_imgs[i].reshape([width_img, height_img])),
                                      global_step=0)

    def valid_epoch(self, loss, psnr, ssim, imgs, step):
        self.writer.add_scalar(tag='valid_epoch/average_loss', scalar_value=loss, global_step=step)
        self.writer.add_scalar(tag='valid_epoch/average_psnr', scalar_value=psnr.mean(), global_step=step)
        self.writer.add_scalar(tag='valid_epoch/average_ssim', scalar_value=ssim.mean(), global_step=step)

        width_img = imgs.shape[1]
        height_img = imgs.shape[2]
        channel = imgs.shape[3]

        for i in range(imgs.shape[0]):
            self.writer.add_scalar(tag='valid_epoch/index_%d_psnr' % i, scalar_value=psnr[i], global_step=step)
            self.writer.add_scalar(tag='valid_epoch/index_%d_ssim' % i, scalar_value=ssim[i], global_step=step)
            for j in range(channel):
                self.writer.add_image(tag='valid_epoch/prediction_index_%d_chaneel_%d' % (i, j),
                                      img_tensor=imgProcess.normalize(imgs[i].reshape([width_img, height_img])),
                                      global_step=step)

    def imgs_valid_init(self, x_imgs, y_imgs):
        width_img = x_imgs.shape[1]
        height_img = x_imgs.shape[2]
        channel = x_imgs.shape[3]

        for i in range(x_imgs.shape[0]):
            for j in range(channel):
                self.writer.add_image(tag='valid_epoch/input_index_%d_chaneel_%d' % (i, j),
                                      img_tensor=imgProcess.normalize(x_imgs[i].reshape([width_img, height_img])),
                                      global_step=0)
                self.writer.add_image(tag='valid_epoch/ground-truth_index_%d/chaneel_%d' % (i, j),
                                      img_tensor=imgProcess.normalize(y_imgs[i].reshape([width_img, height_img])),
                                      global_step=0)

    def config_init(self):
        config_info = open('./config.ini', 'r')
        config_info_string = config_info.read()
        config_info.close()

        self.writer.add_text(tag='config', text_string=config_info_string, global_step=0)
