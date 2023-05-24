import numpy as np
import tiledbsoma as soma
import torch

import cellxgene_census
from cellxgene_census.experimental.ml.pytorch import experiment_dataloader, ExperimentDataPipe

# TODO: Convert this to a notebook


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.2):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=dropout_prob)

    def forward(self, x):
        outputs = self.dropout(self.linear(x))
        return outputs


def train_epoch(model, train_dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    y_true = []
    y_pred = []
    for batch in train_dataloader:
        optimizer.zero_grad()
        X_batch, y_batch = batch

        # exclude the soma_joinid of the y_batch, which is in the first column
        # TODO: allow correct types to be specified to experiment_dataloader()
        y_batch = y_batch[:, 1:].flatten().long()

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        probs = torch.nn.functional.softmax(outputs, 1)
        preds = torch.argmax(probs, axis=1).cpu()

        train_correct += (preds == y_batch.cpu()).sum().item()
        train_total += len(preds)

        loss = loss_fn(outputs, y_batch)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(preds.numpy())

    train_loss /= train_total
    train_accuracy = train_correct / train_total
    return train_loss, train_accuracy


def train(model, model_filename, train_dataloader, device, learning_rate, weight_decay, num_epochs, precision=7):
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        losses.append(round(train_loss, precision))
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.7f} Accuracy {train_accuracy:.4f}")

    torch.save(model.state_dict(), model_filename)

    print("Saved model to file", model_filename)
    return model


def run():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    census = cellxgene_census.open_soma()

    predicted_label = "cell_type"
    obs_value_filter = "tissue_general == 'tongue' and is_primary_data == True"
    var_value_filter = ""

    exp_dp = ExperimentDataPipe(
        census["census_data"]["homo_sapiens"],
        ms_name="RNA",
        layer_name="raw",
        obs_query=soma.AxisQuery(value_filter=(obs_value_filter or None)),
        var_query=soma.AxisQuery(value_filter=(var_value_filter or None)),
        obs_column_names=[predicted_label],
        batch_size=16,
    )

    dp = exp_dp.shuffle(buffer_size=len(exp_dp))
    dp_train, dp_test = dp.random_split(weights={"train": 0.7, "test": 0.3}, seed=RANDOM_SEED)

    dl_train = experiment_dataloader(
        dp_train,
        # >= 1 uses multiprocessing to load data
        num_workers=0,
    )

    pred_field_encoder = exp_dp.obs_encoders()[predicted_label]
    output_dim = len(pred_field_encoder.classes_)
    input_dim = exp_dp.shape[1]

    model = LogisticRegression(input_dim, output_dim).to(device)
    model = train(
        model,
        "model.pt",
        dl_train,
        device,
        learning_rate=1e-4,
        weight_decay=0.0,
        num_epochs=3,
    )
    return model


RANDOM_SEED = 123
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


if __name__ == "__main__":
    run()
