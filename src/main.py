import os
import sys
import json
from notebook_processing.processor import process_notebook, process_notebook_with_metadata_report
from validators.validator import validate_instruction, validate_notebook_schema, extract_notebook_sections_as_dict
from data_loader import template_json

def run_validation(input_json_path: str, output_log_path: str) -> None:
    """Run validation on the input JSON and save results to output path."""
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogues = [data] if isinstance(data, dict) else data
    results = []

    for d_index, dialogue in enumerate(dialogues):
        dialogue_id = dialogue.get("dialogue_metadata", {}).get("dialogue_id", f"dialogue_{d_index}")
        for t_index, turn in enumerate(dialogue["turns"]):
            instructions = turn.get("instructions", {})
            instruction_list = instructions.get("instructions", [])
            all_responses = {k: v for k, v in turn.items() if k.endswith("_response") or k == "response"}

            for label, response in all_responses.items():
                turn_results = []
                for inst in instruction_list:
                    inst_id = inst.get("instruction_id")
                    if not inst_id:
                        continue
                    # Get all kwargs except instruction_id
                    kwargs = {k: v for k, v in inst.items() if k != "instruction_id"}
                    valid, message = validate_instruction(response, inst_id, kwargs, instructions)
                    turn_results.append({
                        "instruction": inst_id,
                        "status": "Passed" if valid else "Failed",
                        "message": message
                    })

                results.append({
                    "dialogue_id": dialogue_id,
                    "turn_index": t_index + 1,
                    "response_type": label,
                    "prompt": turn["prompt"][:100],
                    "results": turn_results
                })

    with open(output_log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Validation complete. Log saved to: {output_log_path}")

def run_batch_processing(input_dir: str, output_base_dir: str) -> None:
    """Process all notebooks in the input directory and validate their outputs."""
    ipynb_files = [f for f in os.listdir(input_dir) if f.endswith(".ipynb")]
    if not ipynb_files:
        print("No .ipynb files found in input folder.")
        return

    for ipynb_file in ipynb_files:
        base_name = os.path.splitext(ipynb_file)[0]
        input_path = os.path.join(input_dir, ipynb_file)
        output_dir = os.path.join(output_base_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Convert notebook to json and get metadata change report
        print(f"\n📘 Processing notebook: {ipynb_file}")
        converted, metadata_report = process_notebook_with_metadata_report(input_path, dialogue_id=base_name)
        converted_path = os.path.join(output_dir, "converted_output.json")
        with open(converted_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        print(f"✅ Converted JSON saved to: {converted_path}")
        
        log_filename = os.path.join(output_dir, "notebook_validation.log")
        notebook = extract_notebook_sections_as_dict(input_path)
        validate_notebook_schema(notebook, template_json, log_filename)

        # Save metadata change report
        metadata_report_path = os.path.join(output_dir, "metadata_change_report.json")
        with open(metadata_report_path, "w", encoding="utf-8") as f:
            json.dump(metadata_report, f, indent=2, ensure_ascii=False)

        # Step 2: Validate and export report
        validation_txt_path = os.path.join(output_dir, "validation_report.json")
        run_validation(converted_path, validation_txt_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    run_batch_processing(input_dir, input_dir) 