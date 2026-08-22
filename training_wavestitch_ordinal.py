import argparse

import torch

from data_utils import PreprocessorOrdinal
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np
import matplotlib.pyplot as plt
from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
import os
from torch.utils.data import DataLoader

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, RossmanSales, PanamaEnergy, QEC', required=True)
    parser.add_argument('-qec_file_path', type=str)
    parser.add_argument('-hierarchical_features', nargs='+', type=str)
    parser.add_argument('-model_path', type=str, default='saved_models/QEC/model_ordinal.pth')
    parser.add_argument('-backbone', type=str, help='Transformer, S4', default='S4')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    parser.add_argument('-hdim', type=int, default=64, help='hidden embedding dimension')
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, help='batch size', default=1024)
    parser.add_argument('-epochs', type=int, default=100, help='training epochs')
    parser.add_argument('-layers', type=int, default=4, help='number of hidden layers')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    parser.add_argument('-stride', type=int, default=1, help='the stride length to shift the training window by')
    parser.add_argument('-num_res_layers', type=int, default=4, help='the number of residual layers')
    parser.add_argument('-res_channels', type=int, default=64, help='the number of res channels')
    parser.add_argument('-skip_channels', type=int, default=64, help='the number of skip channels')
    parser.add_argument('-diff_step_embed_in', type=int, default=32, help='input embedding size diffusion')
    parser.add_argument('-diff_step_embed_mid', type=int, default=64, help='middle embedding size diffusion')
    parser.add_argument('-diff_step_embed_out', type=int, default=64, help='output embedding size diffusion')
    parser.add_argument('-s4_lmax', type=int, default=100)
    parser.add_argument('-s4_dstate', type=int, default=64)
    parser.add_argument('-s4_dropout', type=float, default=0.0)
    parser.add_argument('-s4_bidirectional', type=bool, default=True)
    parser.add_argument('-s4_layernorm', type=bool, default=True)

    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset.lower() == 'qec':
        qec_params = PreprocessorOrdinal.qec_params_from_input(args.qec_file_path, args.hierarchical_features)
        preprocessor = PreprocessorOrdinal(dataset, qec_params)
    else:
        preprocessor = PreprocessorOrdinal(dataset)
    df = preprocessor.df_cleaned
    training_df = df.loc[preprocessor.train_indices]
    test_df = df.loc[preprocessor.test_indices]
    hierarchical_column_indices = training_df.columns.get_indexer(preprocessor.hierarchical_features)
    if (hierarchical_column_indices < 0).any():
        missing = [
            feature for feature, index in zip(preprocessor.hierarchical_features, hierarchical_column_indices)
            if index < 0
        ]
        raise ValueError(f"Unknown hierarchical feature(s): {missing}")

    # QEC shots are independent sequences. Never let a training window cross a
    # shot boundary, even though the cleaned dataframe is stored long-form.
    training_samples = []
    if dataset.lower() == 'qec':
        train_orig = preprocessor.df_orig.loc[preprocessor.train_indices]
        for shot_id in train_orig['shot_id'].dropna().unique():
            shot_indices = train_orig.index[train_orig['shot_id'] == shot_id]
            shot_df = df.loc[shot_indices]
            for i in range(0, len(shot_df) - args.window_size + 1, args.stride):
                training_samples.append(shot_df.iloc[i:i + args.window_size].values)
    else:
        for i in range(0, len(training_df) - args.window_size + 1, args.stride):
            training_samples.append(training_df.iloc[i:i + args.window_size].values)

    if not training_samples:
        raise ValueError(
            f"No training windows were created. Reduce window_size={args.window_size} "
            "or provide QEC shots with enough rounds."
        )

    in_dim = len(training_df.columns)
    out_dim = len(training_df.columns) - len(hierarchical_column_indices)
    training_dataset = MyDataset(from_numpy(np.array(training_samples)).float())
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)
    all_indices = np.arange(len(training_df.columns))

    # Find the indices not in the index_list
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)

    # Convert to an ndarray
    non_hier_cols = np.array(remaining_indices)
    """TRAINING"""
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            timesteps = randint(diffusion_config['T'], size=(batch.shape[0],)).to(device)
            sigmas = normal(0, 1, size=batch.shape).to(device)
            """Forward noising"""
            alpha_bars = diffusion_config['alpha_bars'].to(device)
            coeff_1 = sqrt(alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            coeff_2 = sqrt(1 - alpha_bars[timesteps]).reshape((len(timesteps), 1, 1))
            conditional_mask = np.ones(batch.shape)
            conditional_mask[:, :, non_hier_cols] = 0
            conditional_mask = from_numpy(conditional_mask).float().to(device)
            batch_noised = (1 - conditional_mask) * (coeff_1 * batch + coeff_2 * sigmas) + conditional_mask * batch
            batch_noised = batch_noised.to(device)
            timesteps = timesteps.reshape((-1, 1))
            # timesteps = timesteps.to(device)
            sigmas_predicted = model(batch_noised, timesteps)
            optimizer.zero_grad()
            sigmas_permuted = sigmas[:, :, non_hier_cols].permute((0, 2, 1))
            sigmas_permuted = sigmas_permuted.to(device)
            loss = criterion(sigmas_predicted, sigmas_permuted)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f'epoch: {epoch}, loss: {total_loss}')
    model_path = os.path.abspath(args.model_path)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved WaveStitch model to {model_path}")
