
import os
import sys
import time
import json
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import DACoNModel
from data import DACoNSingleDataset, dacon_single_pad_collate_fn
from utils import (
    move_data_to_device,
    load_config,
    format_time,
    colorize_target_image,
    get_folder_names,
    get_file_names,
    extract_segment,
    extract_color,
    make_inference_data_list,
)


def check_seg_and_color(data_root):
    char_names = get_folder_names(data_root)
    
    for char_name in char_names:
        frame_names = get_file_names(os.path.join(data_root, char_name, "line"))
        ref_frame_names = get_file_names(os.path.join(data_root, char_name, "ref", "gt"))

        for frame_name in frame_names:
            frame_name_no_ext = os.path.splitext(frame_name)[0]
            line_image_path = os.path.join(data_root, char_name, "line", frame_name)
            seg_path = os.path.join(data_root, char_name, "seg", f"{frame_name_no_ext}.png")
            os.makedirs(os.path.join(data_root, char_name, "seg"), exist_ok=True)

            if not(os.path.isfile(seg_path)):
                extract_segment(line_image_path, seg_path)
                
        for frame_name in ref_frame_names:
            frame_name_no_ext = os.path.splitext(frame_name)[0]
            color_image_path = os.path.join(data_root, char_name, "ref", "gt", frame_name)
            line_image_path = os.path.join(data_root, char_name, "ref", "line", frame_name)
            seg_path = os.path.join(data_root, char_name, "ref", "seg", f"{frame_name_no_ext}.png")
            color_json_path = os.path.join(data_root, char_name, "ref", "seg", f"{frame_name_no_ext}.json")
            os.makedirs(os.path.join(data_root, char_name, "ref", "seg"), exist_ok=True)

            if not(os.path.isfile(seg_path)):
                extract_segment(line_image_path, seg_path)
            if not(os.path.isfile(color_json_path)):
                extract_color(color_image_path, seg_path, color_json_path)


def main(arg): 
    config = load_config(arg.config)
    model_path = arg.model
    data_root = arg.data
    inference_start_time = time.time()

    # --- Hyperparameters from config ---
    version = args.version
    if version == None:
        version = config['version']
    num_workers_val = config['datasets']['val']['num_worker']

    batch_size = 1
    save_images = config['val']['save_images']
    save_json = config['val']['save_json']
    config_save_path = config['val'].get('save_path')
    save_path = config_save_path if config_save_path else data_root
    os.makedirs(save_path, exist_ok=True)
    use_dift_only = getattr(args, "use_dift_only", False)
    feature_backbone_only = getattr(args, "feature_backbone_only", False)
    if feature_backbone_only:
        config.setdefault('network', {})
        config['network']['feature_backbone_only'] = (
            feature_backbone_only or config['network'].get('feature_backbone_only', False)
        )
        print("Feature-backbone-only mode enabled (UNet branch disabled).")
    if use_dift_only:
        if feature_backbone_only:
            raise RuntimeError("Cannot enable both DIFT-only and feature-backbone-only inference modes.")
        print("Using DIFT-only segment correspondence mode.")


    device = torch.device("cuda" if torch.cuda.is_available() and config['num_gpu'] > 0 else "cpu")
    print(f"Using device: {device}")

    model = DACoNModel(config['network'], version).to(device)

    print(f"Loading checkpoint {os.path.basename(model_path)}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    
    # --- Dataset and DataLoader setup ---
    print(f"Extracting Segment and Color")
    check_seg_and_color(data_root)

    print("\n--- Start Inference ---")

    model.eval()
    with torch.no_grad():
        char_names = get_folder_names(data_root)

        for char_name in char_names:

            ref_data_list = make_inference_data_list(data_root, char_name, is_ref = True)
            ref_dataset = DACoNSingleDataset(ref_data_list, data_root, is_ref=True, mode = "infer")
            ref_dataloader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_val, collate_fn=dacon_single_pad_collate_fn)

            if use_dift_only and not model.use_dift:
                raise RuntimeError("DIFT-only inference requested but DIFT is disabled in the configuration.")

            all_seg_feats_ref_list = []
            all_seg_colors_ref_list = []

            for i, ref_data in enumerate(ref_dataloader):
                ref_data = move_data_to_device(ref_data, device)
                seg_colors_ref = ref_data['seg_colors']
                seg_feats_ref, _, seg_dift_ref = model._process_single(ref_data['line_image'], ref_data['seg_image'], ref_data["seg_num"])

                feats_ref_current = seg_dift_ref if use_dift_only else seg_feats_ref
                if use_dift_only and feats_ref_current is None:
                    raise RuntimeError("DIFT features were not computed for reference data. Ensure DIFT is enabled.")

                for b in range(batch_size):
                    all_seg_feats_ref_list.append(feats_ref_current[b])
                    all_seg_colors_ref_list.append(seg_colors_ref[b])
                        
                del ref_data, seg_feats_ref
                torch.cuda.empty_cache()
                
            inference_data_list = make_inference_data_list(data_root, char_name, is_ref = False)
            inference_dataset = DACoNSingleDataset(inference_data_list, data_root, is_ref=False, mode = "infer")
            inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_val, collate_fn=dacon_single_pad_collate_fn)

            print(f"\n  Inference on {len(inference_dataset)} samples of {char_name}")

            if not all_seg_feats_ref_list or not all_seg_colors_ref_list:
                raise RuntimeError(f"No reference segments found for {char_name}.")

            all_seg_feats_ref = torch.cat(all_seg_feats_ref_list, dim=0)
            all_seg_colors_ref = torch.cat(all_seg_colors_ref_list, dim=0)

            all_seg_feats_ref = all_seg_feats_ref.unsqueeze(0)
            all_seg_feats_ref = all_seg_feats_ref.repeat(batch_size, 1, 1)

            for i, data in enumerate(inference_dataloader):
                data = move_data_to_device(data, device)
                seg_feats_tgt, _, seg_dift_tgt = model._process_single(data['line_image'], data['seg_image'], data["seg_num"])
                feats_tgt_current = seg_dift_tgt if use_dift_only else seg_feats_tgt
                if use_dift_only and feats_tgt_current is None:
                    raise RuntimeError("DIFT features were not computed for target data. Ensure DIFT is enabled.")

                seg_sim_map = model.get_seg_cos_sim(all_seg_feats_ref.unsqueeze(1), feats_tgt_current.unsqueeze(1))
                seg_sim_map = seg_sim_map.squeeze(1)

                char_name = data["char_name"]
                frame_name = data["frame_name"]
                line_image_tgt = data["line_image"] 
                seg_image_tgt = data["seg_image"] 

                for b in range(batch_size):

                    seg_sim_map_batch = seg_sim_map[b]
                    nearest_patch_indices = torch.argmax(seg_sim_map_batch, dim=-1)
                    
                    color_list_pred = all_seg_colors_ref[nearest_patch_indices]
                    color_list_pred = color_list_pred * 255

                    if save_images:
                        image_pred = colorize_target_image(color_list_pred, line_image_tgt[b], seg_image_tgt[b])
                        folder_path = os.path.join(save_path, char_name[b], "pred")
                        os.makedirs(folder_path, exist_ok=True)
                        file_path  = os.path.join(folder_path, f"{frame_name[b]}.png")
                        image_pred = image_pred.permute(2, 0, 1)
                        save_image(image_pred, file_path)

                    if save_json:
                        folder_path = os.path.join(save_path, char_name[b], "pred")
                        os.makedirs(folder_path, exist_ok=True)
                        json_file_path  = os.path.join(folder_path, f"{frame_name[b]}.json")
                        color_dict = {str(idx + 1): [int(value) for value in color.tolist()] for idx, color in enumerate(color_list_pred)}
                        
                        with open(json_file_path, "w") as json_file:
                            json.dump(color_dict, json_file)

                print(f"  Sample {i+1}/{len(inference_dataset)}", end='\r')
                
            del data, seg_feats_tgt, seg_sim_map
            torch.cuda.empty_cache()

        del all_seg_feats_ref, all_seg_colors_ref
        torch.cuda.empty_cache()

    print("\n--- Inference complete! ---")
    inference_finish_time = time.time()
    print(f"\nTotal time: {format_time(inference_finish_time - inference_start_time)}")


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.append(str(ROOT))

    parser = argparse.ArgumentParser(description="Test the DACoN model.")
    parser.add_argument('--config', type=str, default='configs/inference.yaml', help='Path to the YAML configuration file.')
    parser.add_argument('--model', type=str, default='checkpoints/dacon_v1_1.pth', help='Path to the DACoN weights.')
    parser.add_argument('--data', type=str, default='./inference', help='Root to the Inference images.')
    parser.add_argument('--version', type=str, default=None, help='version of DACoN architecture.')
    parser.add_argument('--use-dift-only', action='store_true', help='Use only DIFT segment features for correspondence.')
    parser.add_argument('--feature-backbone-only', action='store_true', help='Bypass the UNet branch and use feature-backbone descriptors only.')
    args = parser.parse_args()

    main(args)
