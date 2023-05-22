import pandas as pd
import torch
import numpy as np
from anndata import AnnData
from torch.utils.data import Dataset, DataLoader

import cellxgene_census


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
    # p, r, f1, _ = precision_recall_score_support(y_true, y_pred, average=average)
    return train_loss, train_accuracy  # , p, r, f1


def train(
    model, model_filename, train_dataset, device, learning_rate, batch_size, weight_decay, num_epochs, precision=7
):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses = []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        # train_loss, train_accuracy, train_p, train_r, train_f1, train_y_true, train_y_pred = evaluate(model,
        #                                                                                               train_dataloader,
        #                                                                                               loss_fn, device,
        #                                                                                               average='macro')
        losses.append(round(train_loss, precision))

        # print("Epoch", (epoch + 1), ": Train Loss: %.7f Accuracy %.4f Precision: %.4f Recall: %.4f F1: %.4f" % (
        # train_loss, train_accuracy, train_p, train_r, train_f1))
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.7f} Accuracy {train_accuracy:.4f}")

    torch.save(model.state_dict(), model_filename)

    print("Saved model to file", model_filename)
    return model


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_train_test_data(adata: AnnData, pred_field):
    # Normalize anndata raw counts
    # sc.pp.normalize_total(data, layer='raw_counts', target_sum=TARGET_SUM,
    #                       exclude_highly_expressed=True)  # normalized values stored in the raw_counts layer
    # sc.pp.log1p(data, layer='raw_counts')

    # X = data.layers['raw_counts']
    y = adata.obs[pred_field].values

    labels = np.array(list(set(y)))

    y_mapping = {v: i for i, v in enumerate(labels)}
    # y_cat = [y_mapping[i] for i in y]

    # TODO
    data_train, data_test = adata, adata  # train_test_split(data, test_size=test_size, random_state=12, stratify=y_cat)
    X_train_arr = data_train.X.toarray()

    X_train = torch.Tensor(X_train_arr)
    y_train = [y_mapping[i] for i in data_train.obs[pred_field].values]

    X_test_arr = data_test.X.toarray()
    X_test = torch.Tensor(X_test_arr)
    y_test = [y_mapping[i] for i in data_test.obs[pred_field].values]

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    labels = set(y_train)
    labels_cat = np.arange(len(labels))

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset, data_train, data_test, labels_cat, y_mapping


def get_train_test_data(adata: AnnData, pred_field):
    # Normalize anndata raw counts
    # sc.pp.normalize_total(data, layer='raw_counts', target_sum=TARGET_SUM,
    #                       exclude_highly_expressed=True)  # normalized values stored in the raw_counts layer
    # sc.pp.log1p(data, layer='raw_counts')

    # X = data.layers['raw_counts']
    y = adata.obs[pred_field].values

    labels = np.array(list(set(y)))

    y_mapping = {v: i for i, v in enumerate(labels)}
    # y_cat = [y_mapping[i] for i in y]

    # TODO
    data_train, data_test = adata, adata  # train_test_split(data, test_size=test_size, random_state=12, stratify=y_cat)
    X_train_arr = data_train.X.toarray()

    X_train = torch.Tensor(X_train_arr)
    y_train = [y_mapping[i] for i in data_train.obs[pred_field].values]

    X_test_arr = data_test.X.toarray()
    X_test = torch.Tensor(X_test_arr)
    y_test = [y_mapping[i] for i in data_test.obs[pred_field].values]

    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    labels = set(y_train)
    labels_cat = np.arange(len(labels))

    return X_train, y_train, X_test, y_test, train_dataset, test_dataset, data_train, data_test, labels_cat, y_mapping


def get_census_data(num_cells_subsampled, max_samples) -> AnnData:
    census = cellxgene_census.open_soma(
        uri="/Users/atolopko/cxg/cell-census/data/cell-census-small/soma/2023-03-29/soma"
    )
    # census = cellxgene_census.open_soma(census_version='latest')
    data_obs = (
        census["census_data"]["homo_sapiens"]
        .obs.read()
        # value_filter="dataset_id == \"24ec2dc5-3573-4d66-a9e1-25b7dcf43e27\" and is_primary_data == True and assay in [\"10x 3' v1\", \"10x 3' v2\", \"10x 3' v3\", \"10x 3' transcription profiling\", \"10x 5' v1\", \"10x 5' v2\", \"10x 5' transcription profiling\"] and disease == 'normal'")
        .concat()
        .to_pandas()
    )
    if len(data_obs) == 0:
        return None
    census_datasets = (
        census["census_info"]["datasets"]
        .read(column_names=["collection_name", "dataset_title", "dataset_id", "soma_joinid"])
        .concat()
        .to_pandas()
    )
    # census_datasets = census_datasets.set_index("dataset_id")
    # dataset_cell_counts = pd.DataFrame(data_obs[["dataset_id"]].value_counts())
    # dataset_cell_counts = dataset_cell_counts.rename(columns={0: "cell_counts"})
    # dataset_cell_counts = dataset_cell_counts.merge(census_datasets, on="dataset_id")

    data_var = census["census_data"]["homo_sapiens"].ms["RNA"].var.read().concat().to_pandas()
    # presence_matrix = cellxgene_census.get_presence_matrix(census, "Homo sapiens", "RNA")
    # presence_matrix = presence_matrix[dataset_cell_counts.soma_joinid, :]
    # var_somaid = np.nonzero(presence_matrix.sum(axis=0).A1 == presence_matrix.shape[0])[0].tolist()
    # data_var = data_var.query(f"soma_joinid in {var_somaid}")
    # print('There are', data_obs['soma_joinid'].count(), 'num data entries.')

    # if num_cells_subsampled:
    #     data_cell_subsampled_ids = data_obs["soma_joinid"].sample(num_cells_subsampled,
    #                                                               random_state=RANDOM_SEED).tolist()
    # else:
    #     data_cell_subsampled_ids = data_obs["soma_joinid"].tolist()
    #     print(len(data_cell_subsampled_ids), len(set(data_cell_subsampled_ids)))

    # downsampled_soma_join_ids = []
    # data_obs = data_obs[data_obs['soma_joinid'].isin(data_cell_subsampled_ids)]
    # tissue_type_obs_counts = data_obs['tissue_general'].value_counts()
    # for tissue_type, num_obs in zip(tissue_type_obs_counts.index, tissue_type_obs_counts):
    #     # print(count, cell_type)
    #     soma_join_ids = data_obs[data_obs['tissue_general'] == tissue_type]['soma_joinid'].values
    #     if max_samples:
    #         downsampled_join_ids = np.random.choice(soma_join_ids, min(max_samples, len(soma_join_ids)),
    #                                                 replace=False)
    #     else:
    #         downsampled_join_ids = soma_join_ids
    #     downsampled_soma_join_ids.extend(downsampled_join_ids)

    obs_soma_joinids = data_obs["soma_joinid"].to_numpy()
    var_soma_joinids = data_var["soma_joinid"].to_numpy()

    data = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_coords=obs_soma_joinids,
        var_coords=var_soma_joinids,
    )

    print("Learning from", len(data), "cells")
    data.var_names = data.var["feature_name"]
    return data


LEARNING_RATE = 1e-4
NUM_EPOCHS = 1000
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0
RANDOM_SEED = 123
PRED_FIELD = "cell_type"

NUM_DATA_CELLS_SUBSAMPLED = None
MAX_SAMPLES = 50000
LRs = [1e-4]
BATCH_SIZES = [16]
WEIGHT_DECAYS = [0.0]


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    census_adata = get_census_data(NUM_DATA_CELLS_SUBSAMPLED, MAX_SAMPLES)

    (
        X_train,
        y_train,
        X_test,
        y_test,
        train_dataset,
        test_dataset,
        data_train,
        data_test,
        labels_categorical,
        y_mapping,
    ) = get_train_test_data(census_adata, PRED_FIELD)

    input_dim = X_train.shape[1]
    output_dim = len(labels_categorical)

    model = LogisticRegression(input_dim, output_dim).to(device)
    model = train(
        model,
        "model.pt",
        train_dataset,
        device,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=NUM_EPOCHS,
    )
    return model


if __name__ == "__main__":
    main()
