import argparse, torch, csv
from src.datasets import LabelFileDataset
from src.lit_module import LitSingleCell

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",      type=str, required=True)  
    ap.add_argument("--checkpoint",  type=str, required=True)  # ckpt 路径
    ap.add_argument("--arch",        type=str, default="efficientnetb0")
    ap.add_argument("--out_csv",     type=str, default=r"/root/autodl-tmp/projects/mwh/SingleCellProject/outputs/classification/preds.csv")
    ap.add_argument("--img_size",    type=int, default=300)
    args = ap.parse_args()

    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    ds = LabelFileDataset(args.labels, args.img_size, mean, std, None, False)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LitSingleCell({"arch":args.arch,"pretrained":False}, num_classes=2)
    state = torch.load(args.checkpoint, map_location=device)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    try: model.load_state_dict(sd, strict=False)
    except:
        new_sd = {k.replace("core.model.","model.").replace("core.",""):v for k,v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
    model.to(device).eval()

    rows=[]
    with torch.no_grad():
        for (imgs, _labels) in loader:
            logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1)[:,1].cpu().tolist()
            rows.extend(probs)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["image","prob_pos"])
        for (path,_),p in zip(ds.samples, rows):
            w.writerow([path, p])
