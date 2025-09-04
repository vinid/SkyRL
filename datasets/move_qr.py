import json
import shutil
import os
from pathlib import Path

def move_qr_data():
    base_dir = Path(__file__).parent
    qrdata_benchmark_dir = base_dir / "repos" / "QRData" / "benchmark"
    
    data_dir = qrdata_benchmark_dir / "data"
    qrdata_dir = base_dir / "data" / "qrdata"
    qrdata_json_path = qrdata_benchmark_dir / "QRData.json"
    
    data_inside_qrdata = qrdata_dir / "data"
    if data_inside_qrdata.exists():
        raise FileExistsError(f"Directory {data_inside_qrdata} already exists inside qrdata")
    
    qrdata_json_destination = base_dir / "qrdata.json"
    if qrdata_json_destination.exists():
        raise FileExistsError(f"File {qrdata_json_destination} already exists")
    
    if data_dir.exists():
        print(f"Copying {data_dir} to {data_inside_qrdata}")
        shutil.copytree(str(data_dir), str(data_inside_qrdata))
    else:
        print(f"Directory {data_dir} does not exist")
    
    if qrdata_json_path.exists():
        print(f"Processing {qrdata_json_path}")
        with open(qrdata_json_path, 'r') as f:
            original_data = json.load(f)
        
        transformed_data = []
        for entry in original_data:
            # Transform data_files to include full paths like discoverybench
            data_files = entry["data_files"]
            full_paths = [f"/data/qrdata/data/{filename}" for filename in data_files]
            
            transformed_entry = {
                "context": entry["data_description"],
                "question": entry["question"],
                "answer": entry["answer"],
                "data": full_paths,
                "metadata": entry["meta_data"]
            }
            transformed_data.append(transformed_entry)
        
        with open(qrdata_json_destination, 'w') as f:
            json.dump(transformed_data, f, indent=2)
        
        print(f"Transformed JSON saved to {qrdata_json_destination}")
        print(f"Original entries: {len(original_data)}")
        print(f"Transformed entries: {len(transformed_data)}")
    else:
        print(f"File {qrdata_json_path} does not exist")

if __name__ == "__main__":
    move_qr_data()
