import os
import pandas as pd
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.models as models
from PIL import Image

# Custom dataset to load images without labels


class CustomTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(root) if f.endswith(
                ('.png', '.jpg', '.jpeg'))]  # List all images

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]


def main():
    # Define data paths
    train_data_path = "data/train"
    val_data_path = "data/val"
    test_data_path = "data/test"

    transform_train = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.CenterCrop(400),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.CenterCrop(400),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = ImageFolder(
        root=train_data_path,
        transform=transform_train)
    val_dataset = ImageFolder(root=val_data_path, transform=transform_test)
    test_dataset = CustomTestDataset(test_data_path, transform=transform_test)

    # Create DataLoaders
    trainloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8)
    valloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8)
    testloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8)

    net = models.resnext50_32x4d(
        weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)

    num_classes = 100

    # Replace default fully connected Layer, which is 2048 * 1000
    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 1024),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes)
    )
    net = net.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(net.parameters(), lr=25e-5, weight_decay=0.0000001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-20, mode='max')

    # Load pretrained model if you have one
    # net = torch.load('model.pth', weights_only=False)
    # net = net.to('cuda')

    EPOCHS = 50
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0

        net.train()
        for i, (inputs, labels) in enumerate(trainloader):
            # In each batch size, updatae weigths once. Inputs contain 64
            # pictures, and labels contain 64 labels
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()   # 1. Zero out old gradients
            outputs = net(inputs)   # 2. Forward pass
            loss = criterion(outputs, labels)   # 3. Compute loss
            loss.backward()     # 3. Backward pass (compute gradients)
            optimizer.step()    # 4. Update weights

            losses.append(loss.item())
            running_loss += loss.item()

        correct, total = 0, 0
        net.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valloader):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = sum(losses) / len(losses)
        scheduler.step(avg_loss)

        print(
            f"Epoch: {epoch + 1} | Loss: {running_loss:.4f} | Vlidation Accuracy: {100 * (correct / total):.2f}%")

    print('Training Done')

    torch.save(net, 'model.pth')

    # Store results
    results = []
    idx_to_class = {
        value: key for key,
        value in train_dataset.class_to_idx.items()}

    # Predict labels for test images
    with torch.no_grad():
        for images, paths in testloader:
            images = images.to('cuda')
            outputs = net(images)

            _, predicted = torch.max(outputs, 1)

            for path, pred in zip(paths, predicted.cpu().numpy()):
                img_name = os.path.basename(path)
                clean_name = os.path.splitext(img_name)[0]
                pred_class = idx_to_class[pred]
                results.append([clean_name, pred_class])

    # Save predictions to CSV
    df = pd.DataFrame(results, columns=["image_name", "pred_label"])
    df.to_csv("prediction.csv", index=False)
    print("Predictions saved to prediction.csv")


if __name__ == '__main__':
    main()
