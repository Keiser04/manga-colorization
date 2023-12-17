import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import albumentations as albu
import argparse
import datetime
import os
import glob


from utils.utils import open_json, weights_init, weights_init_spectr, generate_mask
from model.models import Colorizer, Generator, Content, Discriminator
from model.extractor import get_seresnext_extractor
from dataset.datasets import TrainDataset, FineTuningDataset
from torch.nn import DataParallel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help = "dataset path")
    parser.add_argument('-ft', '--fine_tuning', dest = 'fine_tuning', action = 'store_true')
    parser.add_argument('-g', '--gpu', dest = 'gpu', action = 'store_true')
    parser.add_argument('-s', '--s_epoch', type=int, default=10, help='save frequency')
    parser.set_defaults(fine_tuning = False)
    parser.set_defaults(gpu = False)
    args = parser.parse_args()
    
    return args

def get_transforms():
    return albu.Compose([albu.RandomCrop(512, 512, always_apply = True), albu.HorizontalFlip(p = 0.5)], p = 1.)

def get_dataloaders(data_path, transforms, batch_size, fine_tuning, mult_number):
    train_dataset = TrainDataset(data_path, transforms, mult_number)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    if fine_tuning:
        finetuning_dataset = FineTuningDataset(data_path, transforms)
        finetuning_dataloader = torch.utils.data.DataLoader(finetuning_dataset, batch_size = batch_size, shuffle = True)
    
    return train_dataloader, finetuning_dataloader

def get_models(device):
    generator = Generator()
    extractor = get_seresnext_extractor()
    colorizer = Colorizer(generator, extractor)
    
    colorizer.extractor_eval()
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        colorizer = nn.DataParallel(colorizer)

    colorizer = colorizer.to(device)
    
    discriminator = Discriminator()
    if torch.cuda.device_count() > 1:
        discriminator = nn.DataParallel(discriminator)
    discriminator = discriminator.to(device)
    
    content = Content('model/vgg16-397923af.pth').eval()
    if torch.cuda.device_count() > 1:
        content = nn.DataParallel(content)
    content = content.to(device)
    
    for param in content.parameters():
        param.requires_grad = False
    
    return colorizer, discriminator, content

def set_weights(colorizer, discriminator):
    if isinstance(colorizer, nn.DataParallel):
        colorizer.module.generator.apply(weights_init)
        colorizer.module.load_extractor_weights(torch.load('model/extractor.pth'))
    else:
        colorizer.generator.apply(weights_init)
        colorizer.load_extractor_weights(torch.load('model/extractor.pth'))
    
    discriminator.apply(weights_init_spectr)

def generator_loss(disc_output, true_labels, main_output, guide_output, real_image, content_gen, content_true, dist_loss = nn.L1Loss(), content_dist_loss = nn.MSELoss(), class_loss = nn.BCEWithLogitsLoss()):    
    sim_loss_full = dist_loss(main_output, real_image)
    sim_loss_guide = dist_loss(guide_output, real_image)
    
    adv_loss = class_loss(disc_output, true_labels)
    
    content_loss = content_dist_loss(content_gen, content_true)
    
    sum_loss = 10 * (sim_loss_full + 0.9 * sim_loss_guide)  + adv_loss + content_loss
    
    return sum_loss
    
def get_optimizers(colorizer, discriminator, generator_lr, discriminator_lr):
    if isinstance(colorizer, nn.DataParallel):
        optimizerG = optim.Adam(colorizer.module.generator.parameters(), lr = generator_lr, betas=(0.5, 0.9))
    else:
        optimizerG = optim.Adam(colorizer.generator.parameters(), lr = generator_lr, betas=(0.5, 0.9))
    
    optimizerD = optim.Adam(discriminator.parameters(), lr = discriminator_lr, betas=(0.5, 0.9))
    
    return optimizerG, optimizerD

def generator_step(inputs, colorizer, discriminator, content,  loss_function, optimizer, device, white_penalty = True):
    for p in discriminator.parameters():
        p.requires_grad = False  
    if isinstance(colorizer, nn.DataParallel):
        for p in colorizer.module.generator.parameters():
            p.requires_grad = True    
    else:
        for p in colorizer.generator.parameters():
            p.requires_grad = True    

    if isinstance(colorizer, nn.DataParallel):
        colorizer.module.generator.zero_grad()
    else:
        colorizer.generator.zero_grad()

    bw, color, hint, dfm = inputs  
    bw, color, hint, dfm = bw.to(device), color.to(device), hint.to(device), dfm.to(device)

    fake, guide = colorizer(torch.cat([bw, dfm, hint], 1))

    logits_fake = discriminator(fake)
    y_real = torch.ones((bw.size(0), 1), device = device)

    content_fake = content(fake)
    with torch.no_grad():
        content_true = content(color)

    generator_loss = loss_function(logits_fake, y_real, fake, guide, color, content_fake, content_true)
    
    if white_penalty:
        mask = (~((color > 0.85).float().sum(dim = 1) == 3).unsqueeze(1).repeat((1, 3, 1, 1 ))).float()
        white_zones = mask * (fake + 1) / 2
        white_penalty = (torch.pow(white_zones.sum(dim = 1), 2).sum(dim = (1, 2)) / (mask.sum(dim = (1, 2, 3)) + 1)).mean()
        
        generator_loss += white_penalty

    generator_loss.backward()

    optimizer.step()
    
    return generator_loss.item()

def decrease_lr(optimizer, rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] /= rate

def set_lr(optimizer, value):
    for param_group in optimizer.param_groups:
        param_group['lr'] = value

def discriminator_step(inputs, colorizer, discriminator, optimizer, device, loss_function = nn.BCEWithLogitsLoss()):
    
    for p in discriminator.parameters():
        p.requires_grad = True 
    if isinstance(colorizer, nn.DataParallel):
        for p in colorizer.module.generator.parameters():
            p.requires_grad = False
    else:
        for p in colorizer.generator.parameters():
            p.requires_grad = False

    discriminator.zero_grad()

    bw, color, hint, dfm = inputs  
    bw, color, hint, dfm = bw.to(device), color.to(device), hint.to(device), dfm.to(device)
    
    y_real = torch.full((bw.size(0), 1), 0.9, device = device)

    y_fake = torch.zeros((bw.size(0), 1), device = device)

    with torch.no_grad():
        fake_color, _ = colorizer(torch.cat([bw, dfm, hint], 1))
        fake_color.detach()

    logits_fake = discriminator(fake_color)
    logits_real = discriminator(color)

    fake_loss = loss_function(logits_fake, y_fake)
    real_loss = loss_function(logits_real, y_real)
    
    discriminator_loss = real_loss + fake_loss

    discriminator_loss.backward()
    optimizer.step()
    
    return discriminator_loss.item()

def save_checkpoint(epoch, colorizer, optimizer):
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    if isinstance(colorizer, nn.DataParallel):
        torch.save({
            'epoch': epoch,
            'model_state_dict': colorizer.module.generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': colorizer.generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

def load_checkpoint(checkpoint_path, colorizer, optimizer):
    checkpoint = torch.load(checkpoint_path)
    if isinstance(colorizer, nn.DataParallel):
        colorizer.module.generator.load_state_dict(checkpoint['model_state_dict'])
    else:
        colorizer.generator.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return epoch

def train(colorizer, discriminator, content, dataloader, epochs, colorizer_optimizer, discriminator_optimizer, lr_decay_epoch=-1, s_epoch=1, device='cpu'):
    start_epoch = 0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Intenta cargar el último punto de control si existe
    try:
        checkpoint_path = max(glob.glob(checkpoint_dir + "/*"), key=os.path.getctime)
        checkpoint = torch.load(checkpoint_path)
        if isinstance(colorizer, nn.DataParallel):
            colorizer.module.generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            colorizer.generator.load_state_dict(checkpoint['model_state_dict'])
        colorizer_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    except ValueError:
        print("No checkpoint found, starting training from scratch.")
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        colorizer = nn.DataParallel(colorizer)
        discriminator = nn.DataParallel(discriminator)
        content = nn.DataParallel(content)

    if isinstance(colorizer, nn.DataParallel):
        colorizer.module.generator.train()
    else:
        colorizer.generator.train()
    discriminator.train()

    disc_step = True

    for epoch in range(start_epoch, start_epoch + epochs):
        if epoch == lr_decay_epoch:
            decrease_lr(colorizer_optimizer, 10)
            decrease_lr(discriminator_optimizer, 10)

        sum_disc_loss = 0
        sum_gen_loss = 0

        for n, inputs in enumerate(dataloader):
            if n % 5 == 0:
                print(datetime.datetime.now().time())
                print('Step : %d Discr loss: %.4f Gen loss : %.4f \n' % (n, sum_disc_loss / (n // 2 + 1), sum_gen_loss / (n // 2 + 1)))

            if disc_step:
                step_loss = discriminator_step(inputs, colorizer, discriminator, discriminator_optimizer, device)
                sum_disc_loss += step_loss
            else:
                step_loss = generator_step(inputs, colorizer, discriminator, content, generator_loss, colorizer_optimizer, device)
                sum_gen_loss += step_loss

            disc_step = disc_step ^ True

        print(datetime.datetime.now().time())
        print('Epoch : %d Discr loss: %.4f Gen loss : %.4f \n' % (epoch, sum_disc_loss / (n // 2 + 1), sum_gen_loss / (n // 2 + 1)))

        if epoch % s_epoch == 0:
            save_checkpoint(epoch, colorizer, gen_optimizer)

def fine_tuning_step(data_iter, colorizer, discriminator, gen_optimizer, disc_optimizer, device, loss_function=nn.BCEWithLogitsLoss()):
            for p in discriminator.parameters():
                p.requires_grad = True
            if isinstance(colorizer, nn.DataParallel):
                for p in colorizer.module.generator.parameters():
                    p.requires_grad = False
            else:
                for p in colorizer.generator.parameters():
                    p.requires_grad = False

            for cur_disc_step in range(5):
                discriminator.zero_grad()

                bw, dfm, color_for_real = data_iter.next()
                bw, dfm, color_for_real = bw.to(device), dfm.to(device), color_for_real.to(device)

                y_real = torch.full((bw.size(0), 1), 0.9, device=device)
                y_fake = torch.zeros((bw.size(0), 1), device=device)

                empty_hint = torch.zeros(bw.shape[0], 4, bw.shape[2], bw.shape[3]).float().to(device)

                with torch.no_grad():
                    fake_color_manga, _ = colorizer(torch.cat([bw, dfm, empty_hint], 1))
                    fake_color_manga.detach()

                logits_fake = discriminator(fake_color_manga)
                logits_real = discriminator(color_for_real)

                fake_loss = loss_function(logits_fake, y_fake)
                real_loss = loss_function(logits_real, y_real)
                discriminator_loss = real_loss + fake_loss

                discriminator_loss.backward()
                disc_optimizer.step()

            for p in discriminator.parameters():
                p.requires_grad = False
            if isinstance(colorizer, nn.DataParallel):
                for p in colorizer.module.generator.parameters():
                    p.requires_grad = True
            else:
                for p in colorizer.generator.parameters():
                    p.requires_grad = True

            if isinstance(colorizer, nn.DataParallel):
                colorizer.module.generator.zero_grad()
            else:
                colorizer.generator.zero_grad()

            bw, dfm, _ = data_iter.next()
            bw, dfm = bw.to(device), dfm.to(device)

            y_real = torch.ones((bw.size(0), 1), device=device)

            empty_hint = torch.zeros(bw.shape[0], 4, bw.shape[2], bw.shape[3]).float().to(device)

            fake_manga, _ = colorizer(torch.cat([bw, dfm, empty_hint], 1))

            logits_fake = discriminator(fake_manga)
            adv_loss = loss_function(logits_fake, y_real)

            generator_loss = adv_loss

            generator_loss.backward()
            gen_optimizer.step()

def fine_tuning(colorizer, discriminator, content, dataloader, iterations, colorizer_optimizer, discriminator_optimizer, data_iter, device='cpu'):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        colorizer = nn.DataParallel(colorizer)
        discriminator = nn.DataParallel(discriminator)
        content = nn.DataParallel(content)

    if isinstance(colorizer, nn.DataParallel):
        colorizer.module.generator.train()
    else:
        colorizer.generator.train()
    discriminator.train()

    disc_step = True

    for n, inputs in enumerate(dataloader):
        if n == iterations:
            return

        if disc_step:
            discriminator_step(inputs, colorizer, discriminator, discriminator_optimizer, device)
        else:
            generator_step(inputs, colorizer, discriminator, content, generator_loss, colorizer_optimizer, device)

        disc_step = disc_step ^ True

        if n % 10 == 5:
            fine_tuning_step(data_iter, colorizer, discriminator, colorizer_optimizer, discriminator_optimizer, device)

if __name__ == '__main__':
    args = parse_args()
    config = open_json('configs/train_config.json')

    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    augmentations = get_transforms()

    train_dataloader, ft_dataloader = get_dataloaders(args.path, augmentations, config['batch_size'], args.fine_tuning, config['number_of_mults'])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        device = torch.device("cuda")
        colorizer, discriminator, content = get_models(device)
        colorizer = DataParallel(colorizer)
        discriminator = DataParallel(discriminator)
        content = DataParallel(content)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        colorizer, discriminator, content = get_models(device)

    set_weights(colorizer, discriminator)

    gen_optimizer, disc_optimizer = get_optimizers(colorizer, discriminator, config['generator_lr'], config['discriminator_lr'])

    train(colorizer, discriminator, content, train_dataloader, config['epochs'], gen_optimizer, disc_optimizer, config['lr_decrease_epoch'], args.s_epoch, device)

    if args.fine_tuning:
        set_lr(gen_optimizer, config["finetuning_generator_lr"])
        fine_tuning(colorizer, discriminator, content, train_dataloader, config['finetuning_iterations'], gen_optimizer, disc_optimizer, iter(ft_dataloader), device)
