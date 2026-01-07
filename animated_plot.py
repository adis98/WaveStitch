import argparse
import torch
from data_utils import Preprocessor
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig
import numpy as np
from torch import from_numpy, optim, nn, randint, normal, sqrt, device, save
from torch.utils.data import DataLoader
import pandas as pd
import os
from metasynth import metadataMask
from matplotlib import pyplot as plt
from copy import deepcopy
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation


def decimal_places(series):
    return series.apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()


def create_pipelined_noise(test_batch, args):
    sampled = torch.normal(0, 1, (args.stride * (test_batch.shape[0] - 1) + args.window_size, test_batch.shape[2]))
    sampled_noise = sampled.unfold(0, args.window_size, args.stride).transpose(1, 2)
    return sampled_noise


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '-d', type=str,
                        help='MetroTraffic, BeijingAirQuality, AustraliaTourism, WebTraffic, StoreItems',
                        default='AustraliaTourism')
    parser.add_argument('-backbone', type=str, help='Transformer, Bilinear, Linear, S4', default='S4')
    parser.add_argument('-beta_0', type=float, default=0.0001, help='initial variance schedule')
    parser.add_argument('-beta_T', type=float, default=0.02, help='last variance schedule')
    parser.add_argument('-timesteps', '-T', type=int, default=200, help='training/inference timesteps')
    parser.add_argument('-hdim', type=int, default=64, help='hidden embedding dimension')
    parser.add_argument('-lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-batch_size', type=int, help='batch size', default=1024)
    parser.add_argument('-layers', type=int, default=4, help='number of hidden layers')
    parser.add_argument('-window_size', type=int, default=32, help='the size of the training windows')
    parser.add_argument('-stride', type=int, default=8, help='the stride length to shift the training window by')
    parser.add_argument('-num_res_layers', type=int, default=4, help='the number of residual layers')
    parser.add_argument('-res_channels', type=int, default=64, help='the number of res channels')
    parser.add_argument('-skip_channels', type=int, default=64, help='the number of skip channels')
    parser.add_argument('-diff_step_embed_in', type=int, default=32, help='input embedding size diffusion')
    parser.add_argument('-diff_step_embed_mid', type=int, default=64, help='middle embedding size diffusion')
    parser.add_argument('-diff_step_embed_out', type=int, default=64, help='output embedding size diffusion')
    parser.add_argument('-s4_lmax', type=int, default=100)
    parser.add_argument('-s4_dstate', type=int, default=64)
    parser.add_argument('-s4_dropout', type=float, default=0.0)
    parser.add_argument('-no_stitch', type=bool, default=False)
    parser.add_argument('-s4_bidirectional', type=bool, default=True)
    parser.add_argument('-s4_layernorm', type=bool, default=True)
    parser.add_argument('-propCycEnc', type=bool, default=False)
    parser.add_argument('-synth_mask', type=str, default='0.5',
                        help="the hierarchy masking type, coarse (C), fine (F), mid (M)")

    args = parser.parse_args()
    dataset = args.dataset
    device = device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocessor = Preprocessor(dataset, args.propCycEnc)
    df = preprocessor.df_cleaned

    #  Add some more samples form the training set as additional context for synthesis
    end = preprocessor.test_indices[-1]
    start = preprocessor.test_indices[0]
    window_cnt = ((end + 1 - args.window_size - start) // args.stride) + 1
    tilde_start = end + 1 - args.window_size - (window_cnt * args.stride)
    additional_indices = start - tilde_start
    test_df = df.loc[preprocessor.train_indices[-additional_indices:] + preprocessor.test_indices]
    test_df_with_hierarchy = preprocessor.cyclicDecode(test_df)
    decimal_accuracy_orig = preprocessor.df_orig.apply(decimal_places).to_dict()
    decimal_accuracy_processed = test_df_with_hierarchy.apply(decimal_places).to_dict()
    decimal_accuracy = {}
    for key in decimal_accuracy_processed.keys():
        decimal_accuracy[key] = decimal_accuracy_orig[key]

    metadata = test_df_with_hierarchy[preprocessor.hierarchical_features_uncyclic]
    rows_to_synth = metadataMask(metadata, args.synth_mask, args.dataset)
    real_df = test_df_with_hierarchy[rows_to_synth]
    df_synth = test_df.copy()
    """Approach 1: Pipeline"""
    test_samples = []
    mask_samples = []
    d_vals = df_synth.values
    m_vals = rows_to_synth.values

    d_vals_tensor = from_numpy(d_vals)
    m_vals_tensor = from_numpy(m_vals)
    windows = d_vals_tensor.unfold(0, args.window_size, args.stride).transpose(1, 2)
    masks = m_vals_tensor.unfold(0, args.window_size, args.stride)
    hierarchical_column_indices = df_synth.columns.get_indexer(preprocessor.hierarchical_features_cyclic)
    in_dim = len(df_synth.columns)
    out_dim = len(df_synth.columns) - len(hierarchical_column_indices)
    test_dataset = MyDataset(windows.float())
    mask_dataset = MyDataset(masks)
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    mask_dataloader = DataLoader(mask_dataset, batch_size=args.batch_size)
    all_indices = np.arange(len(df_synth.columns))
    #
    # # Find the indices not in the index_list
    remaining_indices = np.setdiff1d(all_indices, hierarchical_column_indices)
    #
    # # Convert to an ndarray
    non_hier_cols = np.array(remaining_indices)
    channel = non_hier_cols[0]
    if args.propCycEnc:
        saved_params = torch.load(f'saved_models/{args.dataset}/model_prop.pth', map_location=device)
    else:
        saved_params = torch.load(f'saved_models/{args.dataset}/model.pth', map_location=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(saved_params[name])
            param.requires_grad = True
    model.eval()

    with torch.no_grad():
        synth_tensor = torch.empty(0, test_dataset.inputs.shape[2]).to(device)
        for idx, (test_batch, mask_batch) in enumerate(zip(test_dataloader, mask_dataloader)):
            test_batch = test_batch.to(device)
            mask_batch = mask_batch.to(device)
            test_cpu = test_batch.cpu().numpy()
            mask_cpu = mask_batch.cpu().numpy()
            gt_windows = np.array([test_cpu[j, :, 0] for j in range(3)])
            window = args.window_size
            stride = args.stride
            mask_merged = np.concatenate(
                (mask_cpu[0, :], mask_cpu[1, window - stride:], mask_cpu[2, window - stride:]))
            x_global = np.arange(0, window + 2 * stride)  # long enough to show both windows
            windows_padded = np.array([np.empty(window + 2 * stride) for j in range(3)])
            for j in range(3):
                windows_padded[j, :] = np.nan
                windows_padded[j, stride * j: stride * j + window] = gt_windows[j]

            figs, axs = plt.subplots(3, 1, sharex=True, sharey=True)
            for ax in axs:
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.yaxis.set_visible(False)

            if args.no_stitch:
                figs.suptitle('No stitch loss')
            else:
                figs.suptitle('With stitch loss')
            step_text = figs.text(
                0.5, 0.02,  # FIGURE coordinates
                "",
                ha="center",
                va="bottom",
                fontsize=14,
                color="black"
            )
            lines = []
            for j in range(3):
                axs[j].plot(x_global, windows_padded[j, :], color='blue', label=f'ground truth window')
                ymin, ymax = axs[j].get_ylim()
                for k in range(stride * j, stride * j + window):
                    if mask_merged[k]:
                        axs[j].axvspan(k - 0.5, k + 0.5, color='grey', alpha=0.6, ymin=ymin, ymax=ymax, transform=axs[j].transData)
            x = create_pipelined_noise(test_batch, args).to(device)
            x.requires_grad_()
            x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]
            print(f'batch: {idx} of {len(test_dataloader)}')
            mask_expanded = torch.zeros_like(test_batch, dtype=bool)
            for channel in non_hier_cols:
                mask_expanded[:, :, channel] = mask_batch
            frames = []
            for step in range(diffusion_config['T'] - 1, -1, -1):
                print(f"backward step: {step}")
                times = torch.full(size=(test_batch.shape[0], 1), fill_value=step).to(device)
                alpha_bar_t = diffusion_config['alpha_bars'][step].to(device)
                alpha_bar_t_1 = diffusion_config['alpha_bars'][step - 1].to(device)
                alpha_t = diffusion_config['alphas'][step].to(device)
                beta_t = diffusion_config['betas'][step].to(device)
                sampled_noise = create_pipelined_noise(test_batch, args).to(device)
                conditional_fwd = sqrt(alpha_bar_t) * test_batch + sqrt(1 - alpha_bar_t) * sampled_noise
                if step == diffusion_config['T'] - 1:
                    x[~mask_expanded] = conditional_fwd[~mask_expanded]
                x[:, :, hierarchical_column_indices] = test_batch[:, :, hierarchical_column_indices]

                with torch.enable_grad():
                    epsilon_pred = model(x, times).permute((0, 2, 1))
                    if step > 0:
                        vari = beta_t * ((1 - alpha_bar_t_1) / (1 - alpha_bar_t)) * torch.normal(0, 1,
                                                                                                 size=epsilon_pred.shape).to(
                            device)
                    else:
                        vari = 0.0

                    normal_denoising = create_pipelined_noise(test_batch, args).to(device)
                    normal_denoising[:, :, non_hier_cols] = (x[:, :, non_hier_cols] - (
                            (beta_t / torch.sqrt(1 - alpha_bar_t)) * epsilon_pred)) / torch.sqrt(alpha_t)
                    normal_denoising[:, :, non_hier_cols] += vari
                    masked_binary = mask_batch.int()
                    rolled_x = normal_denoising.roll(1, 0)
                    rolled_x[0, args.stride:, :] = normal_denoising[0, :(args.window_size - args.stride), :]

                    """MSE LOSS"""
                    loss1 = 0.0
                    if not args.no_stitch:
                        loss1 = loss1 + torch.sum((normal_denoising[:, :(args.window_size - args.stride),
                                                   non_hier_cols] - rolled_x[:, args.stride:args.window_size,
                                                                    non_hier_cols]) ** 2, dim=(1, 2))
                    loss2 = torch.sum(~mask_expanded[:, :, non_hier_cols] * (
                            (x[:, :, non_hier_cols] - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / (
                        torch.sqrt(alpha_bar_t)) - test_batch[:, :, non_hier_cols]) ** 2, dim=(1, 2))
                    loss = loss1 + loss2
                    grad = torch.autograd.grad(loss, x, grad_outputs=torch.ones_like(loss))[0]

                x[:, :, non_hier_cols] = normal_denoising[:, :, non_hier_cols]
                eps = -0.1 * grad[:, :, non_hier_cols]
                x[:, :, non_hier_cols] = x[:, :, non_hier_cols] + eps
                x_cpu = x.cpu().numpy()
                x_windows = np.array([x_cpu[j, :, 0] for j in range(3)])
                x_padded = np.array([np.empty(window + 2 * stride) for j in range(3)])
                for j in range(3):
                    x_padded[j, :] = np.nan
                    x_padded[j, stride * j: stride * j + window] = x_windows[j]
                frames.append((x_padded.copy(), step))
                # for j in range(3):
                #     line, = axs[j].plot(x_global, x_padded[j, :], color='green', label=f'gen {j}')
                #     lines.append(line)
                if step == diffusion_config['T'] - 1:  # first iteration
                    # create the 3 green lines now
                    for j in range(3):
                        line, = axs[j].plot(x_global, x_padded[j, :], color='green', label=f'generated window')
                        lines.append(line)
                else:
                    # update the existing line objects
                    for j in range(3):
                        lines[j].set_ydata(x_padded[j, :])
                # for j in range(3):
                #     line, = axs[j].plot(x_global, x_padded[j, :], color='green')
                #     lines.append(line)


            def update(frame_idx):
                x_padded, step = frames[frame_idx]

                # update green lines
                for j in range(3):
                    lines[j].set_ydata(x_padded[j])

                # update text label
                step_text.set_text(f"Denoising step: {step}")

                return lines + [step_text]


            ani = FuncAnimation(
                figs,
                update,
                frames=len(frames),
                interval=100,  # ms per frame
                blit=False
            )
            plt.legend()
            for j in range(3):

                start_x = j * stride
                end_x = j * stride + window
                # Normalize start_x and end_x to the axis coordinates
                x_min_norm = (start_x - x_global[0]) / (x_global[-1] - x_global[0])
                x_max_norm = (end_x - x_global[0]) / (x_global[-1] - x_global[0])
                ymin, ymax = axs[j].get_ylim()
                bar_thickness = (ymax - ymin) * 0.01  # adjust thickness
                axs[j].vlines(j * stride - 0.5, ymin=ymin, ymax=ymax, color='black', linewidth=2)
                axs[j].vlines(j * stride + window - 0.5, ymin=ymin, ymax=ymax, color='black', linewidth=2)
                axs[j].hlines(y=ymax, xmin=start_x-0.5, xmax=end_x-0.5,
                           color='black', linewidth=2)
                axs[j].hlines(y=ymin, xmin=start_x-0.5, xmax=end_x-0.5,
                           color='black', linewidth=2)
            plt.subplots_adjust(top=0.90)
            for ax in axs:
                ax.set_xlim(20, 32)
            # plt.show()
            ani.save("denoising_stitch_zoomed.gif", writer="pillow", fps=10)
            exit()
