import torch
import wandb
from torch import optim, nn
from tqdm import tqdm
from .unet import UNet
from src.load_dataset.data_loading import load_datasets



def train_model(train_loader, val_loader, epochs, lr, device):
    
    wandb.init(
        project="thesis_despeckling",  # Cambia questo nome
        name = "GIRO_DI_PROVA",
        config={
            "epochs": epochs,
            "learning_rate": lr,
            "optimizer": "Adam",
            "loss": "MSELoss",
            "architecture": "UNet",
            "batch_size": train_loader.batch_size,
            "device": device
        }
    )
    model = UNet(in_channels=2, n_classes=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, q_map_in in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            inputs = inputs.to(device)
            q_map_in = q_map_in.to(device)

            optimizer.zero_grad()
            q_map_out = model(inputs)
            q_map_out_norm = torch.sigmoid(q_map_out)
            
            loss = criterion(q_map_out_norm, q_map_in)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            tqdm.write(f'Loss: {loss.item():.4f}')         
        
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, q_map_in in val_loader:
                inputs = inputs.to(device)

                q_map_in = q_map_in.to(device)
                q_map_out = model(inputs)
                q_map_out_norm = torch.sigmoid(q_map_out)
                
                val_loss += criterion(q_map_out_norm, q_map_in)

        val_loss /= len(val_loader)
        
        # Stampa risultati
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # Salva il modello migliore
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../models/best_unet_quality_map.pth')
    
    wandb.finish()
    return model


if __name__ == "__main__":
    LEARINING_RATE = 3e-4
    BATCH_SIZE = 32
    EPOCHS = 20
    CLEAN_PATH = "../../datasets/clean"
    NOISY_PATH = "../../datasets/look4/noisy"
    DENOISED_PATH = "../../datasets/look4/denoised"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader = load_datasets(NOISY_PATH, DENOISED_PATH, CLEAN_PATH)

    model = train_model(train_loader, val_loader, 20, device)