# =============================================================================
# Improved Model with PyTorch and Hugging Face Transformers
# =============================================================================
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict
import numpy as np

# Define constants
MAX_LEN = 256
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Improved BERT-based model
class ImprovedBERT(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedBERT, self).__init__()
        # Load pre-trained BERT model from Hugging Face
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        # Advanced attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=self.bert.config.hidden_size, num_heads=8)

    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Apply advanced attention mechanism
        attention_output, _ = self.attention(bert_output.last_hidden_state, bert_output.last_hidden_state, bert_output.last_hidden_state)
        # Apply dropout to the [CLS] token
        dropout_output = self.dropout(attention_output[:, 0, :])
        # Pass through the classifier to get predictions
        classifier_output = self.classifier(dropout_output)
        return classifier_output

# Custom Dataset class for preparing the data
class BertDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        inputs = self.tokenizer.encode_plus(
            row['doc'],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        label = torch.tensor(row['labels'], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
        }

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare DataLoader
def create_data_loader(dataframe, tokenizer, max_len, batch_size):
    dataset = BertDataset(dataframe, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)

# Assuming train_df and test_df are already defined
train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, TRAIN_BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, VALID_BATCH_SIZE)

# Initialize model
num_classes = len(df_profile.labels.unique())
model = ImprovedBERT(num_classes)

# Define optimizer, scheduler, and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training and evaluation functions
def train_epoch(model, data_loader, criterion, optimizer, scheduler, device):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = criterion(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, criterion, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_data_loader,
        criterion,
        optimizer,
        scheduler,
        device
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        test_data_loader,
        criterion,
        device
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

print('Training complete')
