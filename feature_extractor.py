import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =====================
# CONFIG
# =====================
DATASET_PATH = "datasets/maritime_reid"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# TRANSFORMS (ReID)
# =====================
transform = transforms.Compose([
    transforms.Resize((256, 128)),  # standard ReID size
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# DATASETS
# =====================
train_data = datasets.ImageFolder(
    root=f"{DATASET_PATH}/train",
    transform=transform
)

val_data = datasets.ImageFolder(
    root=f"{DATASET_PATH}/val",
    transform=transform
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

num_classes = len(train_data.classes)

# =====================
# MODEL (ResNet50 ReID)
# =====================
model = models.resnet50(pretrained=True)

# Remove classifier → feature extractor
model.fc = nn.Identity()

model = model.to(device)

# New classifier head (for training only)
classifier = nn.Linear(2048, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(classifier.parameters()),
    lr=LR
)

# =====================
# TRAIN LOOP
# =====================
for epoch in range(EPOCHS):
    model.train()
    classifier.train()

    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        features = model(images)
        outputs = classifier(features)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f}")

# =====================
# SAVE FEATURE EXTRACTOR
# =====================
torch.save(model.state_dict(), "weights/reid_resnet50_boat.pt")
print("✅ ReID model saved")
