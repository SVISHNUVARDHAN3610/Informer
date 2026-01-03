import pickle
import torch
import matplotlib.pyplot as plt


'''set model and valid loder from the trainer.py'''




with open('/kaggle/working/logs/batch-size-32/scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()

for batch in valid_loader:
    pass

x_enc = batch['x_enc'].to(device)
x_dec = batch['x_dec'].to(device)
y_true_normalized = batch['y'].to(device)
means = batch['mean'].to(device) 
stds = batch['std'].to(device)   


with torch.no_grad():
    preds_normalized = model(x_enc, x_dec)
    if preds_normalized.shape[1] > 1:
        preds_normalized = preds_normalized[:, -1, :] 
        y_true_normalized = y_true_normalized[:, -1, :]


preds_real = (preds_normalized * stds.unsqueeze(1)) + means.unsqueeze(1)
y_real = (y_true_normalized * stds.unsqueeze(1)) + means.unsqueeze(1)


plt.figure(figsize=(15, 10))
for i in range(8):
    plt.subplot(4, 2, i+1)
    
    pred_val = preds_real[i].item()
    true_val = y_real[i].item()
    
    color = 'green' if (pred_val > 0 and true_val > 0) or (pred_val < 0 and true_val < 0) else 'red'
    
    plt.bar(['Actual', 'Predicted'], [true_val, pred_val], color=['gray', color])
    plt.title(f"Sample {i}: Actual {true_val:.2f}% vs Pred {pred_val:.2f}%")
    plt.ylabel("% Change")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_results.png')
print("Comparison plot saved to prediction_results.png")
plt.close()


def calculate_directional_accuracy(model, loader, device):
    model.eval()
    correct_direction = 0
    total_samples = 0
    
    print("Calculating full directional accuracy...")
    
    with torch.no_grad():
        for batch in loader:
            x_enc = batch['x_enc'].to(device)
            x_dec = batch['x_dec'].to(device)
            y_true = batch['y'].to(device) 
            
            
            y_pred = model(x_enc, x_dec)
            
            y_pred = y_pred[:, -1, :]
            y_true = y_true[:, -1, :]
            
            pred_sign = torch.sign(y_pred)
            true_sign = torch.sign(y_true)
            
            # Check matches
            matches = (pred_sign == true_sign)
            
            correct_direction += matches.sum().item()
            total_samples += y_true.shape[0]
            
    accuracy = (correct_direction / total_samples) * 100
    print(f"Total Samples: {total_samples}")
    print(f"Correct Directions: {correct_direction}")
    print(f"Final Directional Accuracy: {accuracy:.3f}%")
    
    return accuracy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
acc = calculate_directional_accuracy(model, valid_loader, device)