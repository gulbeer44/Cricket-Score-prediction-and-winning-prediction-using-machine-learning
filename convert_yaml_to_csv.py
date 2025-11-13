# =====================================================
# 🏏 CricSheet YAML → CSV converter (ODI / T20 / Test)
# =====================================================

import os, glob, yaml, pandas as pd
from time import time

BASE = r"C:\IPL FIRST INNINGS SCORE PREDICTION APP"
FOLDERS = ["t20s", "odis", "tests"]

def parse_cricsheet_yaml(file_path, match_type):
    """Extract innings-level summary from CricSheet YAML."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data or "info" not in data or "innings" not in data:
            return pd.DataFrame()

        info = data["info"]
        teams = info.get("teams", [])
        venue = info.get("venue", None)
        if len(teams) != 2:
            return pd.DataFrame()

        innings_records = []

        for inn in data["innings"]:
            inn_name, inn_data = list(inn.items())[0]
            bat_team = inn_data.get("team")
            if not bat_team:
                continue

            # Determine bowling team (opposite of batting)
            bowl_team = [t for t in teams if t != bat_team]
            bowl_team = bowl_team[0] if bowl_team else None

            # Flatten all deliveries
            deliveries = inn_data.get("deliveries", [])
            total_runs = 0
            wickets = 0
            balls = 0

            for d in deliveries:
                ball_key, ball_data = list(d.items())[0]
                runs = ball_data.get("runs", {}).get("total", 0)
                total_runs += runs
                balls += 1
                if "wicket" in ball_data:
                    wickets += 1

            overs = round(balls / 6, 1)

            innings_records.append({
                "bat_team": bat_team,
                "bowl_team": bowl_team,
                "venue": venue,
                "runs": total_runs,    # same as total for first innings summary
                "wickets": wickets,
                "overs": overs,
                "total": total_runs,
                "match_type": match_type
            })

        return pd.DataFrame(innings_records)

    except Exception as e:
        print(f"⚠️ Skipping {os.path.basename(file_path)}: {e}")
        return pd.DataFrame()

# =====================================================
# PROCESS ALL FOLDERS
# =====================================================

if __name__ == "__main__":
    start_all = time()

    for folder in FOLDERS:
        folder_path = os.path.join(BASE, folder)
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder}")
            continue

        print(f"\n📂 Processing {folder} ...")
        files = glob.glob(os.path.join(folder_path, "*.yaml")) + glob.glob(os.path.join(folder_path, "*.yml"))
        if not files:
            print(f"⚠️ No YAML files in {folder}")
            continue

        dfs = []
        for idx, file in enumerate(files, start=1):
            print(f"  [{idx}/{len(files)}] {os.path.basename(file)}", end="  ", flush=True)
            df = parse_cricsheet_yaml(file, folder)
            if not df.empty:
                dfs.append(df)
                print(f"✅ {len(df)} innings")
            else:
                print("⚠️ skipped")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            out_csv = os.path.join(BASE, f"{folder}.csv")
            combined.to_csv(out_csv, index=False)
            print(f"\n💾 Saved {out_csv} ({len(combined)} innings)")
        else:
            print(f"⚠️ No valid innings found in {folder}")

    print(f"\n✅ Completed in {round(time() - start_all, 2)}s total.")
