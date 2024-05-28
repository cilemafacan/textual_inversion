import os
import torch
import argparse
import logging
import time
import torchvision.transforms as T
from utils.util import load_models
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import get_scheduler
from diffusers import StableDiffusionPipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--scheduler', type=str, default='constant')
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--pretrained_model_name', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--json_path', type=str, default='data/image_caption_pairs.json')
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_every', type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename= 'trainer.log')
    logging.info(args)

    tokenizer, noise_scheduler, text_encoder, vae, unet = load_models(args.pretrained_model_name)

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.48145466,0.4578275,0.40821073], std=[0.26862954,0.26130258,0.27577711])])
    dataset = CustomDataset(json_path=args.json_path, 
                            tokenizer=tokenizer, 
                            resolution=args.resolution,
                            transform=transform,
                            img_padding=True)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    optimizer = torch.optim.AdamW(text_encoder.get_input_embeddings().parameters(),
                                  lr=args.learning_rate,
                                  betas=args.betas,
                                  eps=args.eps,
                                  weight_decay=args.weight_decay)
    
    num_training_steps = len(train_loader) * args.num_epochs
    lr_scheduler = get_scheduler(args.scheduler, optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=num_training_steps)

    text_encoder.to(args.device)
    unet.to(args.device)
    vae.to(args.device)

    text_encoder.train()

    start = time.time()
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            
            img,text = batch[0], batch[1]
            latents = vae.encode(img).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=args.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(text)[0].to(dtype=torch.float32)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if epoch % args.save_every == 0:
            pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name,
                                                           text_encoder=text_encoder,
                                                           vae=vae,
                                                           unet=unet,
                                                           tokenizer=tokenizer)
            
            pipe.save_pretrained(args.save_path)

        logging.info(f"Epoch {epoch}/{args.num_epochs} done")
        logging.info(f"Time taken: {time.time() - start}")
        logging.info(f"Loss: {loss.item()}")
        logging.info(f"Learning rate: {lr_scheduler.get_last_lr()[0]}")
        start = time.time()



