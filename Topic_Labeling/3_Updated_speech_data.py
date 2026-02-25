import os
import json
import csv

JSONL_DIR = r"data"
CSV_DIR = "labeled_CSVs"

def load_csv_mapping(csv_path):
    """
    Load a CSV file and return a mapping:
        filename -> {"significant": ..., "topic_extracted": ...}
    """
    mapping = {}
    with open(csv_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = row.get("filename")
            if filename is None:
                continue
            mapping[filename] = {
                "significant": row.get("significant"),
                "topic_extracted": row.get("topic_extracted"),
            }
    return mapping

def process_jsonl_file(jsonl_path, csv_mapping):
    """
    Read a JSONL file, update each row using csv_mapping based on the id field,
    and write back to the same file.
    """
    updated_lines = []

    with open(jsonl_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:

                updated_lines.append(line)
                continue

            row_id = obj.get("id")
            if row_id is not None and row_id in csv_mapping:
                csv_row = csv_mapping[row_id]

                if csv_row["significant"] is not None:
                    obj["significant"] = csv_row["significant"]
                if csv_row["topic_extracted"] is not None:
                    obj["topic_extracted"] = csv_row["topic_extracted"]

            updated_lines.append(json.dumps(obj, ensure_ascii=False))

    with open(jsonl_path, mode="w", encoding="utf-8") as f:
        for line in updated_lines:
            f.write(line + "\n")

def main():

    for filename in os.listdir(JSONL_DIR):
        if not filename.endswith(".jsonl"):
            continue

        jsonl_path = os.path.join(JSONL_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        csv_filename = base_name + ".csv"
        csv_path = os.path.join(CSV_DIR, csv_filename)

        if not os.path.exists(csv_path):
            print(f"CSV file not found for {filename}, expected {csv_filename}")
            continue

        print(f"Processing {jsonl_path} with {csv_path}")

        csv_mapping = load_csv_mapping(csv_path)
        process_jsonl_file(jsonl_path, csv_mapping)

if __name__ == "__main__":
    main()

