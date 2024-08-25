from PIL import Image
import torch
from torch import nn
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler,EulerAncestralDiscreteScheduler, DDPMScheduler
from transformers import CLIPModel, CLIPImageProcessor
from easydict import EasyDict
import yaml
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import torchvision.transforms as transforms
import random
import os
# lightning training code
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import lpips
from utils.EMA import EMA as EMA_callback
from utils.optimizer import get_cosine_schedule_with_warmup
from utils.imgqual_utils import PSNR, SSIM
from utils.loss import L1_Charbonnier_loss
from transformers import CLIPModel, CLIPImageProcessor
from torchmetrics.image.fid import FrechetInceptionDistance
from utils.ip2p_ds import IP2P_dataset
from torchvision.utils import save_image
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置为hf的国内镜像网站

class Instruct_pix2pix(pl.LightningModule):

    def __init__(self, params):
        super(Instruct_pix2pix, self).__init__()
        self.params = params
        self.max_steps = self.params.Trainer.max_steps
        self.save_path = "/hpc2hdd/home/hfeng108/multimodal/instructpix2pix/diffusers/training_ip2p/save_img"
        # self.initlr = self.params.Trainer.initlr

        self.train_datasets = self.params.Trainer.train_datasets
        self.train_batchsize = self.params.Trainer.train_bs
        self.validation_datasets = self.params.Val.val_datasets
        self.val_batchsize = self.params.Val.val_bs
        self.val_crop = True

        self.conditioning_drop_rate = self.params.Model.conditioning_drop_rate
        self.initlr = self.params.Trainer.initlr #initial learning
        self.crop_size = self.params.Trainer.crop_size #random crop size
        self.num_workers = self.params.Trainer.num_workers

        print("Dataset crop size: ", self.crop_size)

        self.loss_f = nn.MSELoss()
        self.noise_scheduler = DDPMScheduler(**self.params.scheduler)
        # used pretrained
        # self.noise_scheduler = DDIMScheduler.from_pretrained(self.params.Model.checkpoint, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(self.params.Model.num_test_timesteps)
        self.vae, self.tokenizer, self.text_encoder, self.unet = self.initialize_model(self.params.Model.checkpoint)
        print('training num:',self.train_dataloader().__len__())
        print('validation num:',self.val_dataloader().__len__())
        self.lpips_fn = lpips.LPIPS(net='alex')
        self.mae = nn.MSELoss()
        self.save_hyperparameters()

        # validation model
        # compute clip similarity
        model_ID = "openai/clip-vit-base-patch16"
        self.guidance_scale = self.params.Model.guidance_scale
        self.image_guidance_scale = self.params.Model.image_guidance_scale
        # data, gt, _, _ = next(iter(loader))
        # data, gt = processor(images=data, return_tensors = "pt"), processor(images=gt, return_tensors = "pt")
        # embedding_a, embedding_b = model.get_image_features(data["pixel_values"]), model.get_image_features(gt["pixel_values"])
        # clip metric
        self.model_ID = "openai/clip-vit-base-patch16"
        self.clip_model, self.clip_processor = self.initialize_clip_metrics(self.model_ID)
        # FID metric
        self.fid = FrechetInceptionDistance(feature = 192).to(self.device)
        self.automatic_optimization = False
    def image_process(self, img):
        img_ls = [] 
        for i in range(img.shape[0]):
            image = img[i, ...]
            image = (image / 2 + 0.5).clamp(0, 1).squeeze()
            image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
            img_ls.append(image)
        return img_ls
    
    def initialize_clip_metrics(self, checkpoint_name):
        model = CLIPModel.from_pretrained(checkpoint_name)
        processor = CLIPImageProcessor.from_pretrained(checkpoint_name)
        # data, gt = processor(images=edited_image, return_tensors = "pt"), processor(images=gt, return_tensors = "pt")
        # embedding_a, embedding_b = model.get_image_features(data["pixel_values"]), model.get_image_features(gt["pixel_values"])
        model.requires_grad_(False)
        return model, processor

    def compute_clip_metrics(self, x, gt):
        x, gt = self.clip_processor(images=x, do_rescale = False,return_tensors = "pt"), self.clip_processor(images=gt, do_rescale = False,return_tensors = "pt")
        x, gt = x["pixel_values"].to(torch.float16).to(self.device), gt["pixel_values"].to(torch.float16).to(self.device)
        # print(x.dtype)
        embedding_gt = self.clip_model.get_image_features(gt)
        embedding_x = self.clip_model.get_image_features(x)
        cos_sim = torch.nn.functional.cosine_similarity(embedding_x, embedding_gt)
        return torch.mean(cos_sim).item()
    
    def initialize_model(self, checkpoint_name):
        vae = AutoencoderKL.from_pretrained(checkpoint_name, subfolder="vae", use_safetensors=True)
        tokenizer = CLIPTokenizer.from_pretrained(checkpoint_name, subfolder = "tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(checkpoint_name, subfolder = "text_encoder", 
                                                    use_safetensors=True)
        unet = UNet2DConditionModel.from_pretrained(checkpoint_name, subfolder = "unet", 
                                                    use_safetensors = True)
        
        # modified unet
        in_channels, out_channel = 8, unet.conv_in.out_channels
        unet.register_to_config(in_channels = in_channels)
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channel, kernel_size=unet.conv_in.kernel_size,
                stride=unet.conv_in.stride, padding=unet.conv_in.padding
            )
            # initialize to zero
            new_conv_in.weight.zero_()
            # copy weight for the first 4 channels
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        return vae, tokenizer, text_encoder, unet
    
    def lpips_score_fn(self,x,gt):
        self.lpips_fn.to(self.device)
        lp_score = self.lpips_fn(
            gt * 2 - 1, x * 2 - 1
        )
        return torch.mean(lp_score).item()
    
    def configure_optimizers(self):
        parameters = [
            {"params": self.unet.parameters()}
        ]
        optimizer = torch.optim.AdamW(parameters, lr=self.initlr,betas=[0.9,0.999], eps = 1e-6, weight_decay=0.01)
        # scheduler2 = get_cosine_schedule_with_warmup(optimizer, 200,self.max_steps) #self.max_steps*0.02,
        
        return [optimizer]
    
    def add_fid(self, pred, gt):
        self.fid.update(pred, real = False)
        self.fid.update(gt, real = True)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        x, gt, prompt, un_cond = batch
        bs, c, h, w = x.shape
        # got latent representation
        latents = self.vae.encode(gt).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        # generate noise
        noise = torch.randn_like(latents).to(self.device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=self.device).long()
        noisy_img = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # get text embedding 
        text_embedding_hidden = self.text_encoder(prompt["input_ids"])[0]
        # 使用众数 mode as image input guidance
        source_imge_embd = self.vae.encode(x).latent_dist.mode()

        # classifier free guidance introduced in the source paper
        generator = torch.Generator(device = self.device).manual_seed(42)

        random_p = torch.rand(bs, generator = generator,device = self.device)
        # configure prompt mask
        prompt_mask = random_p < 2 * self.conditioning_drop_rate
        prompt_mask = prompt_mask.reshape(bs, 1, 1)
        
        # text conditioning  
        null_conditioning = self.text_encoder(un_cond["input_ids"])[0]
        # if mask null condtion embd else text embed
        text_embedding_hidden = torch.where(prompt_mask, null_conditioning, text_embedding_hidden)

        # Sample mask for original images
        image_mask_dtype = source_imge_embd.dtype
        # print(image_mask_dtype)
        image_mask = 1 - (
            (random_p >= self.conditioning_drop_rate).to(image_mask_dtype)
             * (random_p < 3 * self.conditioning_drop_rate).to(image_mask_dtype)
        )
        image_mask = image_mask.reshape(bs, 1, 1, 1)
        # final image conditioning
        source_imge_embd = image_mask * source_imge_embd

        # final guidance input:
        noisy_img = torch.cat([noisy_img, source_imge_embd], dim = 1)
        model_pred = self.unet(noisy_img, timesteps, text_embedding_hidden).sample
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else latents
        loss = self.loss_f(model_pred, target)
        self.manual_backward(loss)
        opt.step()
        # sch = self.lr_schedulers()
        # sch.step()

        with torch.no_grad():
            self.log("train_loss", loss, prog_bar = True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, gt, prompt, un_cond = batch
        bs, c, h, w = x.shape
       
        # text guide
        text_embedding_hidden = self.text_encoder(prompt["input_ids"])[0]
        # uncond text guide
        null_conditioning = self.text_encoder(un_cond["input_ids"])[0]
        # final text embd
        prompt_embd = torch.cat([text_embedding_hidden, null_conditioning, null_conditioning])
        
        # image final guide
        # source guide
        source_imge_embd = self.vae.encode(x).latent_dist.mode()
        un_cond_image =  torch.zeros_like(source_imge_embd)
        final_image_embd = torch.cat([source_imge_embd, un_cond_image, un_cond_image])
        generator = torch.Generator(device = self.device).manual_seed(42)

        # got latent representation
        latents = torch.randn(
        (bs, 4, source_imge_embd.shape[-2], source_imge_embd.shape[-1]),
            generator = generator,
            device = self.device
        )
        latents = latents * self.noise_scheduler.init_noise_sigma
        for t in tqdm(self.noise_scheduler.timesteps):
            # avoid doing three forward step
            # do classifier free guidance
            latent_model_input = torch.cat([latents] * 3)
            latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, timestep = t)
            # concat latent and images latents
            latent_model_input = torch.cat([latent_model_input, final_image_embd], dim = 1)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states = prompt_embd).sample
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond 
                + self.guidance_scale * (noise_pred_text - noise_pred_image)
                + self.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
        print(latents.shape)
        # decode image
        latents = 1 / 0.18215 * latents
        edited_image = self.vae.decode(latents).sample
        edited_image_ls = self.image_process(edited_image.to(torch.float16))
        # save image comparison
        if batch_idx == 0:
            filename = "before_edit_{}.png".format(self.current_epoch)
            save_image(x[2,...],os.path.join(self.save_path, filename))
            filename = "pred_{}.png".format(self.current_epoch)
            edited_image_ls[2].save(os.path.join(self.save_path, filename))
            # save_image(edited_image[:2,...],os.path.join(self.save_path, filename))
            # save gt
            filename = "target_{}.png".format(self.current_epoch)
            save_image(gt[2,...],os.path.join(self.save_path, filename))
        
        cos_sim = self.compute_clip_metrics(edited_image_ls, gt)
        lpips_score = self.lpips_score_fn(edited_image.float(), gt.float())
        self.add_fid((edited_image*255).to(torch.uint8), (gt*255).to(torch.uint8))
        # log metric
        mae = self.mae(edited_image, gt)
        self.log("MSE", mae, sync_dist = True)
        self.log("LPIPS_score", lpips_score, sync_dist = True)
        self.log("Clip_image_sim", cos_sim, sync_dist = True)
        print("Finish first step validation. ")
        return {"LPIPS_score":lpips_score, "Clip_image_sim":cos_sim, "MSE":mae}
        
    def on_validation_epoch_end(self) -> None:
        # return
        # calculate fid
        fid_score = self.fid.compute()
        # clear cache
        self.fid.reset()
        self.log("Epoch FID: ", fid_score, sync_dist = True)
        
    
    def train_dataloader(self):
        
        train_set = IP2P_dataset(self.train_datasets,train=True,size=self.crop_size, tokenizer=self.tokenizer)
        train_loader = DataLoader(train_set, batch_size=self.train_batchsize, shuffle=True, num_workers=self.num_workers)

        return train_loader
    
    def val_dataloader(self):
        val_set = IP2P_dataset(self.validation_datasets,train=False,size=256,crop=True, tokenizer=self.tokenizer)
        val_loader = DataLoader(val_set, batch_size=self.val_batchsize, shuffle=False, num_workers=self.num_workers)

        return val_loader

if __name__=="__main__":
  

    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters = True)
    config_path = r'/hpc2hdd/home/hfeng108/multimodal/instructpix2pix/diffusers/training_ip2p/option/IP2P_edited.yaml'
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    config = EasyDict(params)
    model = Instruct_pix2pix(config)
    checkpoint_callback = ModelCheckpoint(
        monitor='MSE',
        filename='epoch{epoch:02d}-Clip_image_sim-{Clip_image_sim:.3f}-MSE-{MSE:.4f}',
        auto_insert_metric_name=False,   
        every_n_epochs=1,
        save_top_k=5,
        mode = "min",
        save_last=True
        )
    ema_ck = EMA_callback(decay = 0.999)
    output_dir = '/hpc2hdd/home/hfeng108/multimodal/instructpix2pix/diffusers/training_ip2p/lgs/IP2P_edited'
    logger = TensorBoardLogger(name=config.log_name,save_dir = output_dir )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        check_val_every_n_epoch=config.Trainer.check_val_every_n_epoch,
        max_steps=config.Trainer.max_steps,
        accelerator=config.Trainer.accelerator,
        devices=config.Trainer.devices,
        precision=config.Trainer.precision,
        accumulate_grad_batches = config.Trainer.accumulate_grad_batches,
        logger=logger,
        strategy=ddp,
        enable_progress_bar=True,
        log_every_n_steps=config.Trainer.log_every_n_steps,
        callbacks = [checkpoint_callback,lr_monitor_callback, ema_ck]
    )
    ckpt_path= None
    # ckpt_path = "/hpc2hdd/home/hfeng108/multimodal/instructpix2pix/diffusers/training_ip2p/lgs/IP2P_edited/IP2P_edited/version_10/checkpoints/epoch364-Clip_image_sim-0.475-MSE-0.2988.ckpt"
    trainer.fit(model, ckpt_path = ckpt_path)
    # trainer.validate(model, ckpt_path = ckpt_path)