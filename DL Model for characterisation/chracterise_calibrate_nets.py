import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import csv
import json
from datetime import datetime
from skimage import color as ski_colour
import matplotlib.pyplot as plt
from PIL import Image


class CharacteriseNet(nn.Module):
    """
    Network to predict displayed pixel color given background features and foreground RGB.
    """

    def __init__(self, background_dim=768, hidden_dim=128):
        super(CharacteriseNet, self).__init__()

        # Input: background features (N) + foreground RGB (3)
        input_dim = background_dim + 3

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # Output RGB
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )

    def forward(self, background, foreground):
        # Concatenate background features and foreground RGB
        x = torch.cat([background, foreground], dim=1)
        return self.network(x)


class CalibrateNet(nn.Module):
    """
    Network to predict required pixel color given background and desired color.
    """

    def __init__(self, background_dim=768, hidden_dim=128):
        super(CalibrateNet, self).__init__()

        # Input: background features (N) + desired RGB (3)
        input_dim = background_dim + 3

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # Output RGB
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )

    def forward(self, background, desired_color):
        # Concatenate background features and desired RGB
        x = torch.cat([background, desired_color], dim=1)
        return self.network(x)


class ColorDataset(Dataset):
    """
    Dataset for colour prediction tasks.
    """

    def __init__(self, db_csv, bg_features, inds):
        self.num_samples = len(inds)
        self.db_csv = db_csv.loc[inds].reset_index(drop=True)
        self.bg_features = bg_features[inds]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.db_csv.iloc[idx]
        foreground = sample[['R-GT', 'G-GT', 'B-GT']].to_numpy().astype('float32')
        measured = sample[['R-Measured', 'G-Measured', 'B-Measured']].to_numpy().astype('float32')
        return {
            'background': self.bg_features[idx].astype('float32'),
            'foreground': foreground,
            'displayed': measured
        }


#class EarlyStopping:
#    def __init__(self, patience = 10, delta = 0.0001):
#        self.patience = patience
#        self.delta = delta
#        self.best_loss = float('inf')
#        self.counter = 0
#        self.early_stop = False
#        self.best_model_state = None

#    def __call__(self, val_loss, model):
#        if val_loss<self.best_loss - self.delta:
#            self.best_loss = val_loss
#            self.counter = 0
#            self.best_model_state = model.state_dict()
#        else:
#            self.counter+=1
#            if self.counter>=self.patience:
#                self.early_stop=True

def calculate_delta_e(color1, color2):
    """Calculate Delta E between two RGB colors."""
    try:
        # Convert RGB to LAB for Delta E calculation
        color1_np = color1.detach().cpu().numpy()
        color2_np = color2.detach().cpu().numpy()

        color1_np = np.clip(color1_np, 0, 1)
        color2_np = np.clip(color2_np, 0, 1)

        lab1 = ski_colour.rgb2lab(color1_np)
        lab2 = ski_colour.rgb2lab(color2_np)

        delta_e = ski_colour.deltaE_ciede2000(lab1, lab2, input_space="CIE Lab")
        return np.mean(delta_e)
    except:
        # Fallback to simple Euclidean distance if function fails
        return torch.mean(torch.sqrt(torch.sum((color1 - color2) ** 2, dim=1))).item()


def create_color_visualization(colors, title, size=(8, 2)):
    """Create a visualization of colours as a horizontal strip."""
    fig, ax = plt.subplots(1, 1, figsize=size)

    # Create color strip
    color_strip = np.expand_dims(colors, axis=0)
    ax.imshow(color_strip, aspect='auto')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    return fig


def save_checkpoint(epoch, characterise_net, calibrate_net, char_optimizer, cal_optimizer,
                    char_scheduler, cal_scheduler, train_losses, val_losses, checkpoint_dir):
    """Save model checkpoint with all necessary information."""
    checkpoint = {
        'epoch': epoch,
        'characterise_net_state_dict': characterise_net.state_dict(),
        'calibrate_net_state_dict': calibrate_net.state_dict(),
        'char_optimizer_state_dict': char_optimizer.state_dict(),
        'cal_optimizer_state_dict': cal_optimizer.state_dict(),
        'char_scheduler_state_dict': char_scheduler.state_dict(),
        'cal_scheduler_state_dict': cal_scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)

    # Also save the latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path, characterise_net, calibrate_net, char_optimizer,
                    cal_optimizer, char_scheduler, cal_scheduler):
    """Load model checkpoint and restore training state."""
    checkpoint = torch.load(checkpoint_path)

    characterise_net.load_state_dict(checkpoint['characterise_net_state_dict'])
    calibrate_net.load_state_dict(checkpoint['calibrate_net_state_dict'])
    char_optimizer.load_state_dict(checkpoint['char_optimizer_state_dict'])
    cal_optimizer.load_state_dict(checkpoint['cal_optimizer_state_dict'])
    char_scheduler.load_state_dict(checkpoint['char_scheduler_state_dict'])
    cal_scheduler.load_state_dict(checkpoint['cal_scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['train_losses'], checkpoint['val_losses']


def train_epoch(characterise_net, calibrate_net, train_loader, char_optimizer, cal_optimizer,
                device, epoch, writer):
    """Train both networks for one epoch."""
    characterise_net.train()
    calibrate_net.train()

    char_losses = []
    cal_losses = []
    char_delta_e_values = []
    cal_delta_e_values = []

    for batch_idx, batch in enumerate(train_loader):
        background = batch['background'].to(device)
        foreground = batch['foreground'].to(device)
        displayed = batch['displayed'].to(device)

        # Train CharacteriseNet
        char_optimizer.zero_grad()
        char_output = characterise_net(background, foreground)
        char_loss = F.mse_loss(char_output, displayed)
        char_loss.backward()
        char_optimizer.step()

        # Train CalibrateNet
        cal_optimizer.zero_grad()
        cal_output = calibrate_net(background, char_output.detach())
        cal_loss = F.mse_loss(cal_output, foreground)
        cal_loss.backward()
        cal_optimizer.step()

        # Calculate Delta E
        char_delta_e = calculate_delta_e(char_output, displayed)
        cal_delta_e = calculate_delta_e(cal_output, foreground)

        char_losses.append(char_loss.item())
        cal_losses.append(cal_loss.item())
        char_delta_e_values.append(char_delta_e)
        cal_delta_e_values.append(cal_delta_e)

        # Log first batch visualization
        if batch_idx == 0:
            with torch.no_grad():
                # Create color visualizations for tensorboard
                sample_size = min(32, len(foreground))

                # CharacteriseNet visualization
                char_expected = displayed[:sample_size].cpu().numpy()
                char_predicted = char_output[:sample_size].cpu().numpy()

                char_exp_fig = create_color_visualization(char_expected, 'CharacteriseNet Expected')
                char_pred_fig = create_color_visualization(char_predicted,
                                                           'CharacteriseNet Predicted')

                writer.add_figure(f'CharacteriseNet/Expected_Colors', char_exp_fig, epoch)
                writer.add_figure(f'CharacteriseNet/Predicted_Colors', char_pred_fig, epoch)

                # CalibrateNet visualization
                cal_expected = foreground[:sample_size].cpu().numpy()
                cal_predicted = cal_output[:sample_size].cpu().numpy()

                cal_exp_fig = create_color_visualization(cal_expected, 'CalibrateNet Expected')
                cal_pred_fig = create_color_visualization(cal_predicted, 'CalibrateNet Predicted')

                writer.add_figure(f'CalibrateNet/Expected_Colors', cal_exp_fig, epoch)
                writer.add_figure(f'CalibrateNet/Predicted_Colors', cal_pred_fig, epoch)

                plt.close('all')  # Close figures to save memory

    return (np.mean(char_losses), np.mean(cal_losses),
            np.mean(char_delta_e_values), np.mean(cal_delta_e_values))


def validate_epoch(characterise_net, calibrate_net, val_loader, device):
    """Validate both networks."""
    characterise_net.eval()
    calibrate_net.eval()

    char_losses = []
    cal_losses = []
    char_delta_e_values = []
    cal_delta_e_values = []

    with torch.no_grad():
        for batch in val_loader:
            background = batch['background'].to(device)
            foreground = batch['foreground'].to(device)
            displayed = batch['displayed'].to(device)

            # CharacteriseNet validation
            char_output = characterise_net(background, foreground)
            char_loss = F.mse_loss(char_output, displayed)

            # CalibrateNet validation
            cal_output = calibrate_net(background, char_output)
            cal_loss = F.mse_loss(cal_output, foreground)

            # Calculate Delta E
            char_delta_e = calculate_delta_e(char_output, displayed)
            cal_delta_e = calculate_delta_e(cal_output, foreground)

            char_losses.append(char_loss.item())
            cal_losses.append(cal_loss.item())
            char_delta_e_values.append(char_delta_e)
            cal_delta_e_values.append(cal_delta_e)

    return (np.mean(char_losses), np.mean(cal_losses),
            np.mean(char_delta_e_values), np.mean(cal_delta_e_values))


def main():
    parser = argparse.ArgumentParser(description='Train CharacteriseNet and CalibrateNet')
    parser.add_argument('--db_path', type=str, default=None, help='Dataset CSV file')
    parser.add_argument('--bg_path', type=str, default=None, help='Background features')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation split')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--scheduler_step', type=int, default=30, help='Scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Scheduler gamma')

    args = parser.parse_args()

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logs_dir = os.path.join(run_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets and dataloaders

    # making the numebr of samples for train and validation
    db_csv = pd.read_csv(args.db_path)
    db_num_sampels = len(db_csv)
    samples_ind = np.arange(db_num_sampels)
    np.random.shuffle(samples_ind)
    val_cut_off = int(db_num_sampels * args.val_split)
    val_inds = samples_ind[:val_cut_off]
    train_inds = samples_ind[val_cut_off:]


    # reading the background
    bg_features = np.load(args.bg_path)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    bg_features = scaler.fit_transform(bg_features)

    pca = PCA(n_components=128)
    bg_features = pca.fit_transform(bg_features)


    train_dataset = ColorDataset(db_csv, bg_features, train_inds)
    val_dataset = ColorDataset(db_csv, bg_features, val_inds)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create models
    background_dim = bg_features.shape[1]
    characterise_net = CharacteriseNet(background_dim, args.hidden_dim).to(device)
    calibrate_net = CalibrateNet(background_dim, args.hidden_dim).to(device)
    #es_char = EarlyStopping(patience=15)
    #es_cal = EarlyStopping(patience=15)

    # Create optimizers
    char_optimizer = optim.Adam(characterise_net.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    cal_optimizer = optim.Adam(calibrate_net.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    # Create schedulers
    char_scheduler = optim.lr_scheduler.StepLR(char_optimizer, step_size=args.scheduler_step,
                                               gamma=args.scheduler_gamma)
    cal_scheduler = optim.lr_scheduler.StepLR(cal_optimizer, step_size=args.scheduler_step,
                                              gamma=args.scheduler_gamma)

    # Setup tensorboard
    writer = SummaryWriter(logs_dir)

    # Setup CSV logging
    csv_path = os.path.join(run_dir, 'training_log.csv')
    csv_headers = [
        'epoch', 'char_lr', 'cal_lr',
        'char_train_loss', 'char_val_loss', 'char_train_delta_e', 'char_val_delta_e',
        'cal_train_loss', 'cal_val_loss', 'cal_train_delta_e', 'cal_val_delta_e'
    ]

    # Initialize training state
    start_epoch = 0
    train_losses = {'char': [], 'cal': []}
    val_losses = {'char': [], 'cal': []}

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        start_epoch, train_losses, val_losses = load_checkpoint(
            args.resume, characterise_net, calibrate_net,
            char_optimizer, cal_optimizer, char_scheduler, cal_scheduler
        )
        start_epoch += 1

    # Create CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)

    print("Starting training...")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Train
        char_train_loss, cal_train_loss, char_train_delta_e, cal_train_delta_e = train_epoch(
            characterise_net, calibrate_net, train_loader, char_optimizer, cal_optimizer,
            device, epoch, writer
        )

        # Validate
        char_val_loss, cal_val_loss, char_val_delta_e, cal_val_delta_e = validate_epoch(
            characterise_net, calibrate_net, val_loader, device
        )

        #Early stopping
        #es_char(char_val_loss, characterise_net)
        #if es_char.early_stop:
        #    print("Early stopping characteriseNet at epoch", epoch)
        #    break

            
        #es_cal(cal_val_loss, calibrate_net)
        #if es_cal.early_stop:
        #    print("Early stopping calibrateNet at epoch", epoch)
        #    break

        # Update schedulers
        char_scheduler.step()
        cal_scheduler.step()

        # Log to tensorboard
        writer.add_scalars('Loss/CharacteriseNet', {
            'Train': char_train_loss,
            'Validation': char_val_loss
        }, epoch)

        writer.add_scalars('Loss/CalibrateNet', {
            'Train': cal_train_loss,
            'Validation': cal_val_loss
        }, epoch)

        writer.add_scalars('DeltaE/CharacteriseNet', {
            'Train': char_train_delta_e,
            'Validation': char_val_delta_e
        }, epoch)

        writer.add_scalars('DeltaE/CalibrateNet', {
            'Train': cal_train_delta_e,
            'Validation': cal_val_delta_e
        }, epoch)

        writer.add_scalars('Learning_Rate', {
            'CharacteriseNet': char_optimizer.param_groups[0]['lr'],
            'CalibrateNet': cal_optimizer.param_groups[0]['lr']
        }, epoch)

        # Log to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                epoch, char_optimizer.param_groups[0]['lr'], cal_optimizer.param_groups[0]['lr'],
                char_train_loss, char_val_loss, char_train_delta_e, char_val_delta_e,
                cal_train_loss, cal_val_loss, cal_train_delta_e, cal_val_delta_e
            ])

        # Print progress
        print(
            f"CharacteriseNet - Train Loss: {char_train_loss:.6f}, Val Loss: {char_val_loss:.6f}, "
            f"Train ΔE: {char_train_delta_e:.3f}, Val ΔE: {char_val_delta_e:.3f}")
        print(f"CalibrateNet - Train Loss: {cal_train_loss:.6f}, Val Loss: {cal_val_loss:.6f}, "
              f"Train ΔE: {cal_train_delta_e:.3f}, Val ΔE: {cal_val_delta_e:.3f}")

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            epoch, characterise_net, calibrate_net, char_optimizer, cal_optimizer,
            char_scheduler, cal_scheduler, train_losses, val_losses, checkpoint_dir
        )
        print(f"Checkpoint saved: {checkpoint_path}")
        print("-" * 80)

    #characterise_net.load_state_dict(es_char.best_model_state)
    #calibrate_net.load_state_dict(es_cal.best_model_state)
    print("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()
