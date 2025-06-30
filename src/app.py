import streamlit as st
import os
import json
from pathlib import Path
import tempfile
from main import run_validation, run_batch_processing
from validators.validator import validate_instruction, check_contradicting_instructions, analyze_instruction_statuses_by_turn
import requests
from data_loader import conflict_dict

st.set_page_config(
    page_title="Turing Amazon Task Parser VIF",
    page_icon="📊",
    layout="wide"
)


def call_nova_api(user_content, system_content="You are a chatbot", temperature=0.7, seed=42, top_p=1, top_k=40, max_tokens=1000):
    url = "https://kong.turing.com/api/llm-gateway"
    headers = {
        "x-api-key": os.getenv("TURING_API_KEY"),
        "x-api-gw-key": os.getenv("TURING_API_GW_KEY"),
        "Authorization": os.getenv("TURING_AUTH_TOKEN"),
        "Content-Type": "application/json"
    }
    payload = {
        "modelName": "us.amazon.nova-premier-v1:0",
        "provider": "Amazon",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        "params": {
            "temperature": temperature,
            "seed": seed,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens
        },
        "images": []
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code in (200, 201):
        data = response.json()
        return data["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def main():
    st.title("Turing Amazon Task Parser VIF")
    st.markdown("Process and validate Jupyter notebooks containing Turing Amazon task data.")
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Validation (Single Turn)",
        "Validation (Batch/Notebook)",
        "Nova (Conversation)",
        "Nova (Batch/Notebook)",
    ])
    with tab1:
        show_single_cell_validation()
    with tab2:
        show_batch_processing()
    with tab3:
        st.subheader("Nova Model: Conversation Test & Validation")
        st.markdown("Test a prompt against the Nova model and validate the result.")
        show_nova_single_turn()
    with tab4:
        st.subheader("Nova Model: Batch Notebook Validation")
        st.markdown("Upload one or more Jupyter notebooks. Each will be run as a sequential conversation against Nova, and each turn will be validated.")
        show_nova_batch_multi()

def show_batch_processing():
    st.header("Notebook Processing and Validation")
    st.markdown("Process and validate a single or multiple Jupyter notebooks.")

    uploaded_files = st.file_uploader(
        "Upload Jupyter notebooks",
        type=['ipynb'],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process Notebooks"):
            with st.spinner("Processing notebooks..."):
                # Create a temporary directory for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    # Process the notebooks
                    run_batch_processing(temp_dir, temp_dir)

                    # Display results
                    st.success("Processing complete!")
                    
                    # Show results for each file
                    for uploaded_file in uploaded_files:
                        base_name = Path(uploaded_file.name).stem
                        result_dir = os.path.join(temp_dir, base_name)
                        
                        if os.path.exists(result_dir):
                            st.subheader(f"Validation Report for {uploaded_file.name}\n")
                            
                            st.subheader("REPORT")
                            notebook_validation = os.path.join(result_dir, "notebook_validation.log")
                            if os.path.exists(notebook_validation):
                                with open(notebook_validation, 'r', encoding='utf-8') as f:
                                    log_content = f.read()
                                log_content = log_content.split('\n')
                                initial_check = True if 'True' in log_content else False
                                st.text('\n'.join(log_content[:-2]))

                            # Display validation report
                            validation_path = os.path.join(result_dir, "validation_report.json")
                            if os.path.exists(validation_path):
                                with open(validation_path, 'r', encoding='utf-8') as f:
                                    validation_data = json.load(f)
                                task_data = analyze_instruction_statuses_by_turn(validation_data)

                                st.subheader("Classification Summary")
                                for line in task_data['text']:
                                    st.markdown(f"- {line}")
                                st.markdown(f'Task Classification: {task_data["classification"]}')

                                if initial_check and not task_data['task_fail']:
                                    st.markdown(f"✅ PRELIMINARY CHECKS PASSED")
                                else:
                                    st.markdown(f"❌ PRELIMINARY CHECKS FAILED")

                                st.subheader("Results Per Turn")
                                st.table(task_data['results_per_turn'])
                                st.subheader("Detailed report")
                                st.json(validation_data)
                            # Display metadata change report
                            metadata_report_path = os.path.join(result_dir, "metadata_change_report.json")
                            if os.path.exists(metadata_report_path):
                                st.markdown("#### Metadata Change Report")
                                with open(metadata_report_path, 'r', encoding='utf-8') as f:
                                    metadata_report = json.load(f)
                                st.json(metadata_report)

def show_single_cell_validation():
    st.header("Single Cell Validation")
    st.markdown("""
    Validate a single cell's assistant response and instructions.
    
    The instructions should follow this JSON schema:
    ```json
    {
      "metadata": ["add"],
      "instructions": [
        {
          "instruction_id": "",
          "kwarg1_name": "kwarg1_value",
          "kwarg2_name": "kwarg2_value"
        }
      ]
    }
    ```
    """)

    # Input for Assistant Response
    assistant_response = st.text_area(
        "Assistant Response",
        height=200,
        help="Paste the assistant's response text here",
        key="1"
    )

    # Input for Instructions JSON
    instructions_json = st.text_area(
        "Instructions JSON",
        height=200,
        help="Paste the instructions JSON following the schema above",
        key="2"
    )

    if st.button("Validate Cell", type="primary"):
        if not assistant_response or not instructions_json:
            st.error("Please provide both Assistant Response and Instructions JSON")
            return

        with st.spinner("Validating cell..."):
            try:
                # Parse instructions JSON
                instructions = json.loads(instructions_json)
                
                # Validate schema
                if not isinstance(instructions, dict):
                    st.error("Instructions must be a JSON object")
                    return
                    
                if "metadata" not in instructions or "instructions" not in instructions:
                    st.error("Instructions must contain 'metadata' and 'instructions' fields")
                    return
                    
                if not isinstance(instructions["metadata"], list):
                    st.error("'metadata' must be a list")
                    return
                    
                if not isinstance(instructions["instructions"], list):
                    st.error("'instructions' must be a list")
                    return
                    
                for instruction in instructions["instructions"]:
                    if not isinstance(instruction, dict):
                        st.error("Each instruction must be an object")
                        return
                    if "instruction_id" not in instruction:
                        st.error("Each instruction must have an 'instruction_id' field")
                        return
                    
                contradiction_errors = check_contradicting_instructions(instructions["instructions"])
                if contradiction_errors:
                    st.error("Contradicting instructions found: " + str(contradiction_errors))
                    return

                # Validate each instruction
                results = []
                for instruction in instructions["instructions"]:
                    inst_id = instruction["instruction_id"]
                    # Remove instruction_id from kwargs
                    kwargs = {k: v for k, v in instruction.items() if k != "instruction_id"}
                    valid, message = validate_instruction(assistant_response, inst_id, kwargs, instructions)
                    results.append({
                        "instruction": inst_id,
                        "status": "Passed" if valid else "Failed",
                        "message": message
                    })

                # Display results
                st.success("Validation complete!")
                st.json({
                    "response": assistant_response[:100] + "..." if len(assistant_response) > 100 else assistant_response,
                    "results": results
                })

            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {str(e)}")
            except Exception as e:
                st.error(f"Error during validation: {str(e)}")

    # Add button to show contradicting pairs
    if st.button("Show Contradicting Instruction Pairs"):
        pairs = []
        for instr, conflicting_list in conflict_dict.items():
            if conflicting_list:
                for conflict in conflicting_list:
                    pairs.append({"Instruction 1": instr, "Instruction 2": conflict})
            else:
                pairs.append({"Instruction 1": instr, "Instruction 2": ""})
        st.markdown("#### Contradicting Instruction Pairs")
        st.table(pairs)

def show_nova_single_turn():
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []
    # Display all previous turns
    remove_turn_idx = None
    for idx, turn in enumerate(st.session_state["conversation"]):
        st.markdown(f"**Turn {idx+1}**")
        st.text_area(f"User Prompt {idx+1}", value=turn["prompt"], key=f"prompt_{idx}", disabled=True)
        st.text_area(f"Instructions JSON {idx+1}", value=turn["instructions_json"], key=f"instructions_{idx}", disabled=True)
        st.code(turn["nova_response"], language=None)
        # Display validation report for previous turns
        if "validation_report" in turn:
            st.markdown("**Validation Report:**")
            st.json(turn["validation_report"])
        if st.button(f"Remove Turn {idx+1}"):
            remove_turn_idx = idx
    if remove_turn_idx is not None:
        st.session_state["conversation"].pop(remove_turn_idx)
        st.rerun()
    # Add new turn (only one at a time, not appended until Run Nova is clicked)
    st.markdown("---")
    st.markdown("**Add New Turn**")
    new_prompt = st.text_area("User Prompt", height=100, key="new_prompt")
    new_instructions_json = st.text_area("Instructions JSON", height=200, help="Paste the instructions JSON following the schema above", key="new_instructions")
    if st.button("Run Nova & Validate Conversation", key="nova_multi"):
        if not new_prompt or not new_instructions_json:
            st.error("Please provide both User Prompt and Instructions JSON for the new turn.")
            return
        # Build conversation context
        messages = []
        for prev_turn in st.session_state["conversation"]:
            messages.append({"role": "user", "content": prev_turn["prompt"]})
            messages.append({"role": "assistant", "content": prev_turn["nova_response"]})
        messages.append({"role": "user", "content": new_prompt})
        # Call Nova with full context
        try:
            instructions = json.loads(new_instructions_json)
        except Exception as e:
            st.error(f"Invalid JSON in new turn: {e}")
            return
        with st.spinner("Calling Nova model for new turn with context..."):
            context_prompt = "\n".join([m["content"] for m in messages])
            nova_response = call_nova_api(context_prompt)
        st.markdown(f"**Nova Model Response for Turn {len(st.session_state['conversation'])+1}:**")
        st.code(nova_response, language=None)
        # Validate
        validation_report = None
        try:
            if not isinstance(instructions, dict):
                st.error(f"Instructions in new turn must be a JSON object")
            elif "metadata" not in instructions or "instructions" not in instructions:
                st.error(f"Instructions in new turn must contain 'metadata' and 'instructions' fields")
            elif not isinstance(instructions["metadata"], list):
                st.error(f"'metadata' must be a list in new turn")
            elif not isinstance(instructions["instructions"], list):
                st.error(f"'instructions' must be a list in new turn")
            else:
                results = []
                for instruction in instructions["instructions"]:
                    if not isinstance(instruction, dict) or "instruction_id" not in instruction:
                        st.error(f"Each instruction in new turn must be an object with 'instruction_id'")
                        continue
                    inst_id = instruction["instruction_id"]
                    kwargs = {k: v for k, v in instruction.items() if k != "instruction_id"}
                    valid, message = validate_instruction(nova_response, inst_id, kwargs, instructions)
                    results.append({
                        "instruction": inst_id,
                        "status": "Passed" if valid else "Failed",
                        "message": message
                    })
                validation_report = {
                    "response": nova_response[:100] + "..." if len(nova_response) > 100 else nova_response,
                    "results": results
                }
                st.success(f"Validation complete for new turn!")
                st.markdown("**Validation Report:**")
                st.json(validation_report)
        except Exception as e:
            st.error(f"Validation error in new turn: {e}")
        # Append the new turn to the conversation, including validation report
        st.session_state["conversation"].append({
            "prompt": new_prompt,
            "instructions_json": new_instructions_json,
            "nova_response": nova_response,
            "validation_report": validation_report
        })
        st.rerun()

def show_nova_batch_multi():
    uploaded_files = st.file_uploader(
        "Upload Jupyter notebooks",
        type=["ipynb"],
        accept_multiple_files=True,
        key="nova_batch_multi"
    )
    if uploaded_files:
        from notebook_processing.processor import process_notebook
        for uploaded_file in uploaded_files:
            st.markdown(f"---\n### Results for `{uploaded_file.name}`")
            with st.spinner(f"Processing `{uploaded_file.name}` and running Nova model for each turn..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        notebook_data = process_notebook(file_path)
                except Exception as e:
                    st.error(f"Failed to process notebook `{uploaded_file.name}`: {e}")
                    continue

                
                results = []
                conversation = []
                for turn in notebook_data.get("turns", []):
                    prompt = turn.get("prompt", "")
                    instructions = turn.get("instructions", {})
                    if not prompt or not instructions:
                        continue
                    # Build conversation context
                    messages = []
                    for prev in conversation:
                        messages.append({"role": "user", "content": prev["prompt"]})
                        messages.append({"role": "assistant", "content": prev["nova_response"]})
                    messages.append({"role": "user", "content": prompt})
                    context_prompt = "\n".join([m["content"] for m in messages])
                    try:
                        nova_response = call_nova_api(context_prompt)
                    except Exception as e:
                        nova_response = f"Nova error: {e}"
                    turn_result = {"prompt": prompt, "instructions": instructions, "nova_response": nova_response, "validation": []}
                    # Validate
                    try:
                        if not isinstance(instructions, dict):
                            turn_result["validation"].append({"error": "Instructions must be a JSON object"})
                        elif "instruction_change" not in instructions or "instructions" not in instructions:
                            turn_result["validation"].append({"error": "Instructions must contain 'instruction_change' and 'instructions' fields"})
                        elif not isinstance(instructions["instruction_change"], list):
                            turn_result["validation"].append({"error": "'instruction_change' must be a list"})
                        elif not isinstance(instructions["instructions"], list):
                            turn_result["validation"].append({"error": "'instructions' must be a list"})
                        else:
                            for instruction in instructions["instructions"]:
                                if not isinstance(instruction, dict) or "instruction_id" not in instruction:
                                    turn_result["validation"].append({"error": "Each instruction must be an object with 'instruction_id'"})
                                    continue
                                inst_id = instruction["instruction_id"]
                                kwargs = {k: v for k, v in instruction.items() if k != "instruction_id"}
                                valid, message = validate_instruction(nova_response, inst_id, kwargs, instructions)
                                turn_result["validation"].append({
                                    "instruction": inst_id,
                                    "status": "Passed" if valid else "Failed",
                                    "message": message
                                })
                    except Exception as e:
                        turn_result["validation"].append({"error": f"Validation error: {e}"})
                    results.append(turn_result)
                    conversation.append({"prompt": prompt, "nova_response": nova_response})
                st.success(f"Batch Nova validation complete for `{uploaded_file.name}`!")
                for i, res in enumerate(results):
                    st.markdown(f"#### Turn {i+1}")
                    st.markdown(f"**Prompt:** {res['prompt']}")
                    st.markdown(f"**Nova Response:** {res['nova_response']}")
                    st.json(res["validation"])

if __name__ == "__main__":
    main() 