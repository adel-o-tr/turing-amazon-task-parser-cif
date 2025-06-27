import re
import string
import json
from typing import Dict, List, Tuple, Any
import copy
import json
import re
from data_loader import conflict_dict
from collections import defaultdict

# Map of expected kwargs for each instruction ID
EXPECTED_ARGUMENTS = {
    "change_case:all_caps": [],
    "change_case:lowercase": [],
    "change_case:alternating": [],
    "change_case:first_letter_cap": [],
    "change_case:capital_word_frequency": ["capital_relation", "capital_frequency"],
    "change_case:lowercase_word_frequency": ["lowercase_relation", "lowercase_frequency"],
    "change_case:all_caps_target": ["target_string"],
    "change_case:lowercase_target": ["target_string"],
    "change_case:alternating_target": ["target_string"],
    "change_case:first_letter_cap_target": ["target_string"],
    "detectable_content:number_placeholders": ["relation", "num_placeholders"],
    "detectable_content:postscript": ["postscript_marker"],
    "detectable_format:json_format": [],
    "detectable_format:multiple_sections": ["section_splitter", "relation", "num_sections"],
    "detectable_format:numbered_list": ["relation", "num_numbered_items"],
    "detectable_format:number_bullet_lists": ["relation", "num_bullets"],
    "detectable_format:title": [],
    "keywords:existence": ["keywords"],
    "keywords:frequency": ["keyword", "relation", "frequency"],
    "keywords:forbidden_words": ["forbidden_words"],
    "keywords:letter_frequency": ["letter", "let_relation", "let_frequency"],
    "punctuation:no_comma": [],
    "length_constraints:number_characters": ["relation", "num_chars"],
    "length_constraints:number_words": ["relation", "num_words"],
    "length:max_word_count": ["max_words"],
    "startend:start_checker": ["start_phrase"],
    "startend:end_checker": ["end_phrase"],
    "startend:wrap_checker": ["wrap_phrase"],
    "startend:quotation": []
}

def is_strict_alternating(word: str) -> bool:
    """Check if a word has strictly alternating case."""
    letters = [c for c in word if c.isalpha()]
    return all(letters[i].isupper() != letters[i + 1].isupper() for i in range(len(letters) - 1))

def char_frequency(response: str, char: str) -> int:
    """Count frequency of a character in response."""
    return response.count(char)

def count_numbered_items(response: str) -> int:
    """Count number of numbered items in response."""
    return len(re.findall(r'^\s*\d+\.', response, re.MULTILINE))

def count_bullet_points(response: str) -> int:
    """Count number of bullet points in response."""
    return len(re.findall(r'^[*-•]\s', response, re.MULTILINE))

def count_placeholders(response: str) -> int:
    """Count number of placeholders in response."""
    return len(re.findall(r'\[.*?\]', response))

def count_all_caps_words(response: str) -> int:
    """Count number of all-caps words in response."""
    return sum(1 for w in response.split() if w.isupper())

def count_lowercase_words(response: str) -> int:
    """Count number of lowercase words in response."""
    return sum(1 for w in response.split() if w.islower())

def word_frequency(response: str, word: str) -> int:
    """Count frequency of a word in response."""
    words = re.findall(r'[^\s]+', response.lower())
    return words.count(word.lower())

def keyword_frequency(response: str, keyword: str) -> int:
    """Count frequency of a keyword in response, ensuring it's a full word or phrase."""
    pattern = r'\b' + re.escape(keyword.strip()) + r'\b'
    return len(re.findall(pattern, response, flags=re.IGNORECASE))

def validate_instruction(response: str, inst_type: str, kwargs: Dict[str, Any], all_instructions: Dict = None) -> Tuple[bool, str]:
    """Validate a response against a specific instruction type and its kwargs."""
    try:
        if inst_type == "change_case:all_caps":
            return (response.isupper(), "No error" if response.isupper() else "Response is not all uppercase.")

        if inst_type == "change_case:lowercase":
            return (response.islower(), "No error" if response.islower() else "Response is not all lowercase.")

        if inst_type == "change_case:alternating":
            valid = all(is_strict_alternating(w) for w in response.split() if w.isalpha())
            return (valid, "No error" if valid else "Response is not strictly alternating caps.")

        if inst_type == "change_case:first_letter_cap":
            valid = all(w.istitle() for w in response.split() if w.isalpha())
            return (valid, "No error" if valid else "Not all words are first-letter capitalized.")

        if inst_type == "change_case:capital_word_frequency":
            count = count_all_caps_words(response)
            rel, val = kwargs['capital_relation'], kwargs['capital_frequency']
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} all-cap words, found {count}.")

        if inst_type == "change_case:lowercase_word_frequency":
            count = count_lowercase_words(response)
            rel, val = kwargs['lowercase_relation'], kwargs['lowercase_frequency']
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} lowercase words, found {count}.")

        if "_target" in inst_type:
            target = kwargs["target_string"].strip().lower()
            target_escaped = re.escape(target)
            pattern = rf'\b{target_escaped}\b'
            matches = re.findall(pattern, response, re.IGNORECASE)

            if not matches:
                return (False, f"Target '{target}' not found in response.")

            for match in matches:
                raw_text = match.strip('"').strip("'")
                if inst_type == "change_case:all_caps_target" and not raw_text.isupper():
                    return (False, f"'{raw_text}' should be ALL CAPS.")
                elif inst_type == "change_case:lowercase_target" and not raw_text.islower():
                    return (False, f"'{raw_text}' should be all lowercase.")
                elif inst_type == "change_case:alternating_target" and not is_strict_alternating(raw_text):
                    return (False, f"'{raw_text}' is not in alternating caps.")
                elif inst_type == "change_case:first_letter_cap_target" and not raw_text.istitle():
                    return (False, f"'{raw_text}' is not first-letter capitalized.")

            return (True, "No error")

        if inst_type == "detectable_content:number_placeholders":
            count = count_placeholders(response)
            rel, val = kwargs["relation"], kwargs["num_placeholders"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} placeholders, found {count}.")

        if inst_type == "detectable_content:postscript":
            marker = kwargs.get("postscript_marker", "PS:").strip()
            lines = response.strip().splitlines()
            for line in reversed(lines):
                if line.strip():
                    last_line = line.strip()
                    break
            else:
                last_line = ""

            has_postscript = last_line.startswith(marker) and len(last_line) > len(marker)
            return (
                has_postscript,
                "No error" if has_postscript else f"Postscript must start with '{marker}' and contain content. Found: '{last_line}'"
            )

        if inst_type == "detectable_format:json_format":
            try:
                json_part = response[response.find("{"):response.rfind("}")+1]
                json.loads(json_part)
                return (True, "No error")
            except:
                return (False, "Response is not valid JSON format.")

        if inst_type == "detectable_format:multiple_sections":
            splitter = kwargs.get("section_splitter", "").strip()
            rel = kwargs.get("relation")
            val = kwargs.get("num_sections")
            pattern = rf"^\s*[#>*\-]*\s*{re.escape(splitter)}\s+\d+\b"
            sections = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
            valid = eval(f"{len(sections)} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} sections, found {len(sections)}.")

        if inst_type == "detectable_format:numbered_list":
            count = count_numbered_items(response)
            rel, val = kwargs["relation"], kwargs["num_numbered_items"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} numbered items, found {count}.")

        if inst_type == "detectable_format:number_bullet_lists":
            count = count_bullet_points(response)
            rel, val = kwargs["relation"], kwargs["num_bullets"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} bullet points, found {count}.")

        if inst_type == "detectable_format:title":
            lines = response.strip().splitlines()
            found_title = any(line.strip().startswith("<<") and line.strip().endswith(">>") for line in lines)
            return (
                found_title,
                "No error" if found_title else "Title not wrapped in << >> on any line."
            )

        if inst_type == "keywords:existence":
            missing = [kw for kw in kwargs["keywords"] if keyword_frequency(response, kw) == 0]
            return (not missing, "No error" if not missing else f"Missing keyword(s): {missing}")

        if inst_type == "keywords:frequency":
            keyword = kwargs["keyword"].strip().lower()
            count = keyword_frequency(response, keyword)
            rel = kwargs["relation"]
            val = kwargs["frequency"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (
                valid,
                "No error" if valid else f"Expected {rel} {val} of '{keyword}', found {count}."
            )

        if inst_type == "keywords:forbidden_words":
            present = [w for w in kwargs["forbidden_words"] if keyword_frequency(response, w)]
            return (not present, "No error" if not present else f"Forbidden words found: {present}")

        if inst_type == "keywords:letter_frequency":
            letter = kwargs["letter"].lower()
            count = response.lower().count(letter)
            rel, val = kwargs["let_relation"], kwargs["let_frequency"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (
                valid,
                "No error" if valid else f"Expected {rel} {val} '{letter}' (case-insensitive), found {count}."
            )

        if inst_type == "punctuation:no_comma":
            return (',' not in response, "No error" if ',' not in response else "Commas found in response.")

        if inst_type == "length_constraints:number_characters":
            count = len(response.strip())
            rel, val = kwargs["relation"], kwargs["num_chars"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} characters, found {count}.")

        if inst_type == "length_constraints:number_words":
            count = len(re.compile(r'\b(?=\S*[A-Za-z0-9])\S+\b').findall(response))
            rel, val = kwargs["relation"], kwargs["num_words"]
            valid = eval(f"{count} {'>=' if rel == 'at least' else '==' if rel == 'equal to' else '<'} {val}")
            return (valid, "No error" if valid else f"Expected {rel} {val} words, found {count}.")

        if inst_type == "startend:start_checker":
            starts_correctly = response.lstrip(string.punctuation + " ").lower().startswith(kwargs.get("start_phrase", "").lower())
            return (
                starts_correctly,
                "No error" if starts_correctly else "Response does not start with required phrase."
            )

        if inst_type == "startend:end_checker":
            required = kwargs["end_phrase"].strip()
            # Check if required phrase ends with punctuation
            ends_with_punctuation = required[-1] in string.punctuation if required else False
            
            # Get the actual end of the response
            actual_words = response.lstrip(string.punctuation).strip().split()
            if not actual_words:
                return (False, "Empty response")
                
            # If required phrase ends with punctuation, we need exact match
            if ends_with_punctuation:
                actual_phrase = " ".join(actual_words[-len(required.split()):])
                if actual_phrase.lower() != required.lower():
                    return (
                        False,
                        f"End phrase mismatch: expected '{required}', but found '{actual_phrase}'"
                    )
            else:
                # If no punctuation, strip trailing punctuation and whitespace
                actual_phrase = " ".join(actual_words).rstrip(string.punctuation + " ")[-len(required):]
                if actual_phrase.lower() != required.lower():
                    return (
                        False,
                        f"End phrase mismatch: expected '{required}', but found '{actual_phrase}'"
                    )
            return (True, "No error")

        if inst_type == "startend:wrap_checker":
            wrap = kwargs["wrap_phrase"]
            return (response.strip().startswith(wrap) and response.strip().endswith(wrap),
                    "No error" if response.strip().startswith(wrap) else f"Not wrapped with: {wrap}")

        if inst_type == "startend:quotation":
            return (response.strip().startswith('"') and response.strip().endswith('"'),
                    "No error" if response.strip().startswith('"') else "Response not wrapped in double quotes.")

    except Exception as e:
        return (False, f"Validation error: {str(e)}")

    return (True, "No error")

def check_contradicting_instructions(instructions_list: List[Dict]) -> List[Dict]:
    """Check for contradicting instruction IDs in the list (order-insensitive)."""
    errors, seen_pairs = set(), set()
    # Collect all instruction IDs
    ids = {inst["instruction_id"] for inst in instructions_list if isinstance(inst, dict) and "instruction_id" in inst}

    # Check each instruction for conflicts
    for instr_id in ids:
        for conflicting_id in conflict_dict.get(instr_id, []):
            pair = frozenset([instr_id, conflicting_id])
            if conflicting_id in ids and pair not in seen_pairs:
                errors.add(f"{instr_id} and {conflicting_id} are contradicting")
                seen_pairs.add(pair)
    return errors

def validate_instruction_schema(instructions: Dict) -> List[Dict]:
    """Validate the schema of instructions against expected arguments and check for contradicting instructions."""
    mismatches = []
    
    # Validate metadata field
    metadata = instructions.get("metadata", [])
    if not isinstance(metadata, list):
        mismatches.append({
            "field": "metadata",
            "expected": "list",
            "actual": type(metadata).__name__
        })
    
    # Validate instructions array
    instructions_list = instructions.get("instructions", [])
    if not isinstance(instructions_list, list):
        mismatches.append({
            "field": "instructions",
            "expected": "list",
            "actual": type(instructions_list).__name__
        })
        return mismatches

    # Check for contradicting instructions
    contradiction_errors = check_contradicting_instructions(instructions_list)
    mismatches.extend(contradiction_errors)


    # Validate each instruction object
    for i, inst in enumerate(instructions_list):
        if not isinstance(inst, dict):
            mismatches.append({
                "instruction_index": i,
                "expected": "dict",
                "actual": type(inst).__name__
            })
            continue

        # Check for required instruction_id field
        if "instruction_id" not in inst:
            mismatches.append({
                "instruction_index": i,
                "missing_field": "instruction_id"
            })
            continue

        # Validate kwargs against expected arguments
        expected_args = set(EXPECTED_ARGUMENTS.get(inst["instruction_id"], []))
        actual_args = set(k for k in inst.keys() if k != "instruction_id")

        if expected_args != actual_args:
            mismatches.append({
                "instruction": inst["instruction_id"],
                "expected_args": sorted(expected_args),
                "actual_args": sorted(actual_args)
            })

    return mismatches 

def extract_notebook_sections_as_dict(ipynb_path):
    with open(ipynb_path, 'r', encoding='utf-8') as file:
        notebook_data = json.load(file)

    result = defaultdict(list)

    for cell in notebook_data.get('cells', []):
        if cell.get('cell_type') != 'markdown':
            continue

        content = ''.join(cell.get('source', [])).strip()
        split_lines = content.splitlines()
        if split_lines[0] == '# Metadata':
            result['task_metadata'].append(content)
            continue

        match = re.search(r'\*\*\[([\w.]+)]\*\*', split_lines[0])
        title = match.group(1)

        result[title].append('\n'.join(split_lines[1:]))

    return result


def validate_notebook_schema(notebook, template_json, log_filename):
    logs = []
    try:
        dict_turn_metadata = turn_metadata_json_to_dict(notebook['turn_metadata'])
        correct_turn_metadata = compare_consecutive_metadata_items(dict_turn_metadata)
        
        conflicting_instructions = find_conflicting_instructions(dict_turn_metadata)
        issues_in_keys_against_template = validate_keys_against_template(template_json, dict_turn_metadata)
        issues_in_instruction_kwargs_datatype = validate_instruction_kwargs_datatype(dict_turn_metadata)

        logs.append(f'CONFLICTING INSTRUCTIONS FOUND - {conflicting_instructions}')
        logs.append(f'INSTRUCTION ARGUMENT MISMATCHES IN TURN JSON - {issues_in_keys_against_template}')
        logs.append(f'VALIDATING JSON SCHEMA - {issues_in_instruction_kwargs_datatype}')

        i, flag = 1, False
        for t, f in zip(correct_turn_metadata, dict_turn_metadata):
            if t['metadata'] != f['metadata']:
                logs.append(f"TURN {i} METADATA SHOULD BE {t['metadata']}, BUT IS {f['metadata']}")
                flag = True
            i += 1
        if not flag:
            logs.append("TURN METADATA IS CORRECT")
            if any([conflicting_instructions, issues_in_keys_against_template, issues_in_instruction_kwargs_datatype]):
                logs.append('False')
            else:
                logs.append('True')
        else:
            logs.append('False')
    except Exception as e:
        logs.append(f'Some error occurred while validating the notebook - {e}')
    finally:
        with open(log_filename, "w", encoding="utf-8") as f:
            f.writelines(line + '\n' for line in logs)


def turn_metadata_json_to_dict(turn_metadata):
    parsed_json_metadata = []
    for item in turn_metadata:
        # Extract the JSON block between triple backticks
        match = re.search(r"```(?:\w+)?\n(.*?)```", item, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            # Parse the JSON string into a dictionary
            data = json.loads(json_str)
            data['metadata'] = set(data['metadata'])
            parsed_json_metadata.append(data)
        else:
            raise "No JSON found in item."
    return parsed_json_metadata


def compare_consecutive_metadata_items(dict_turn_metadata):
    def to_dict(instructions):
        return {instr['instruction_id']: instr for instr in instructions}

    updated = []

    for idx, current_turn in enumerate(dict_turn_metadata):
        if idx == 0:
            updated.append(copy.deepcopy(current_turn))  # Keep the first as-is
            continue

        prev_instr = to_dict(dict_turn_metadata[idx - 1]['instructions'])
        curr_instr = to_dict(current_turn['instructions'])

        metadata = set()

        # Check for additions and modifications
        for instr_id, instr in curr_instr.items():
            if instr_id not in prev_instr:
                print(idx + 1, 'Added', instr_id)
                metadata.add("add")
            elif instr != prev_instr[instr_id]:
                print(idx + 1, 'Modified', instr_id)
                metadata.add("modify")

        # Check for removals
        for instr_id in prev_instr:
            if instr_id not in curr_instr:
                print(idx + 1, 'Removed', instr_id)
                metadata.add("remove")

        # Avoid duplicates
        current_copy = copy.deepcopy(current_turn)
        current_copy['metadata'] = metadata
        updated.append(current_copy)

    return updated


def find_conflicting_instructions(dict_turn_metadata):
    conflicts_found = []

    for data in dict_turn_metadata:
        instruction_ids = {instr["instruction_id"] for instr in data.get("instructions", [])}
        current_conflicts = []

        for instr_id in instruction_ids:
            if instr_id in conflict_dict:
                for conflicting_id in conflict_dict[instr_id]:
                    if conflicting_id in instruction_ids:
                        pair = tuple(sorted((instr_id, conflicting_id)))
                        if pair not in current_conflicts:
                            current_conflicts.append(pair)

        if current_conflicts:
            conflicts_found.append(current_conflicts)

    return conflicts_found


def validate_keys_against_template(template_json, dict_turn_metadata):
    # Map template instruction_id to expected key set
    template_keys = {
        instr["instruction_id"]: set(instr.keys())
        for instr in template_json.get("instructions", [])
    }
    idx, res = 1, []

    for input_json in dict_turn_metadata:
        mismatches = {}

        # Check each instruction in input_json
        for instr in input_json.get("instructions", []):
            instr_id = instr.get("instruction_id")
            input_keys = set(instr.keys())

            if instr_id not in template_keys:
                mismatches[instr_id] = {
                    "error": "instruction_id not in template"
                }
            elif input_keys != template_keys[instr_id]:
                mismatches[instr_id] = {
                    "missing_keys": list(template_keys[instr_id] - input_keys),
                    "extra_keys": list(input_keys - template_keys[instr_id])
                }
        res.append({f"TURN {idx}": mismatches}) if mismatches else ''
        idx += 1
    return res


def validate_instruction_kwargs_datatype(dict_turn_metadata):
    def is_valid_str(val):
        return isinstance(val, str)

    def is_valid_int(val):
        return isinstance(val, int)

    def is_valid_list_str(val):
        return isinstance(val, list) and all(isinstance(i, str) for i in val)

    def is_valid_relation(val):
        return val in valid_relations

    valid_relations = {"at least", "equal to", "less than"}
    turn, issues = 1, []

    for data in dict_turn_metadata:
        errors = []

        # Check metadata
        if not isinstance(data.get("metadata"), set) or not all(isinstance(item, str) for item in data["metadata"]):
            errors.append("metadata must be a list of strings.")

        # Check instructions
        instructions = data.get("instructions", [])
        if not isinstance(instructions, list):
            errors.append("instructions must be a list.")
            return errors

        for idx, inst in enumerate(instructions):
            if not isinstance(inst, dict):
                errors.append(f"Instruction at index {idx} is not a dict.")
                continue

            iid = inst.get("instruction_id")
            if not iid or not isinstance(iid, str):
                errors.append(f"Missing or invalid instruction_id at index {idx}")
                continue

            def add_error(field, expected_type):
                errors.append(f"{iid}: '{field}' must be {expected_type}")

            # Validate per instruction_id
            if iid == "length_constraints:number_characters":
                if not is_valid_relation(inst.get("relation")):
                    add_error("relation", "one of 'at least', 'equal to', 'less than'")
                if not is_valid_int(inst.get("num_chars")):
                    add_error("num_chars", "int")

            elif iid == "keywords:existence":
                if not is_valid_list_str(inst.get("keywords")):
                    add_error("keywords", "list of str")

            elif iid == "detectable_format:numbered_list":
                if not is_valid_relation(inst.get("relation")):
                    add_error("relation", "valid relation")
                if not is_valid_int(inst.get("num_numbered_items")):
                    add_error("num_numbered_items", "int")

            elif iid == "keywords:frequency":
                if not is_valid_str(inst.get("keyword")):
                    add_error("keyword", "str")
                if not is_valid_relation(inst.get("relation")):
                    add_error("relation", "valid relation")
                if not is_valid_int(inst.get("frequency")):
                    add_error("frequency", "int")

            elif iid == "length_constraints:number_words":
                if not is_valid_relation(inst.get("relation")):
                    add_error("relation", "valid relation")
                if not is_valid_int(inst.get("num_words")):
                    add_error("num_words", "int")

            elif iid in {
                "change_case:all_caps_target",
                "change_case:lowercase_target",
                "startend:wrap_checker",
                "change_case:alternating_target",
                "startend:end_checker",
                "change_case:first_letter_cap_target",
                "detectable_content:postscript",
                "startend:start_checker"
            }:
                if not is_valid_str(
                        inst.get("target_string") or inst.get("wrap_phrase") or inst.get("end_phrase") or inst.get(
                            "start_phrase") or inst.get("postscript_marker")):
                    add_error("target_string/wrap_phrase/etc", "str")

            elif iid == "keywords:forbidden_words":
                if not is_valid_list_str(inst.get("forbidden_words")):
                    add_error("forbidden_words", "list of str")

            elif iid == "change_case:lowercase_word_frequency":
                if not is_valid_relation(inst.get("lowercase_relation")):
                    add_error("lowercase_relation", "valid relation")
                if not is_valid_int(inst.get("lowercase_frequency")):
                    add_error("lowercase_frequency", "int")

            elif iid == "keywords:letter_frequency":
                if not is_valid_str(inst.get("letter")):
                    add_error("letter", "str")
                if not is_valid_relation(inst.get("let_relation")):
                    add_error("let_relation", "valid relation")
                if not is_valid_int(inst.get("let_frequency")):
                    add_error("let_frequency", "int")

            elif iid == "change_case:capital_word_frequency":
                if not is_valid_relation(inst.get("capital_relation")):
                    add_error("capital_relation", "valid relation")
                if not is_valid_int(inst.get("capital_frequency")):
                    add_error("capital_frequency", "int")

            elif iid == "detectable_format:multiple_sections":
                if not is_valid_str(inst.get("section_splitter")):
                    add_error("section_splitter", "str")
                if not is_valid_relation(inst.get("relation")):
                    add_error("relation", "valid relation")
                if not is_valid_int(inst.get("num_sections")):
                    add_error("num_sections", "int")

            elif iid == "detectable_format:number_bullet_lists":
                if not is_valid_relation(inst.get("relation")):
                    add_error("relation", "valid relation")
                if not is_valid_int(inst.get("num_bullets")):
                    add_error("num_bullets", "int")

            elif iid == "detectable_content:number_placeholders":
                if not is_valid_relation(inst.get("relation")):
                    add_error("relation", "valid relation")
                if not is_valid_int(inst.get("num_placeholders")):
                    add_error("num_placeholders", "int")

            else:
                # for simple instructions with just instruction_id
                if len(inst) > 1:
                    errors.append(f"{iid}: should not contain incorrect/extra fields")

        issues.append({f'TURN {turn}': errors}) if errors else ''
        turn += 1
    return issues

def analyze_instruction_statuses_by_turn(data):
    results_per_turn, frontier_fail_rates = [], []
    task_fail, nova_fail, resp = False, None, []

    for item in data:
        turn_index = item.get('turn_index')
        response_type = item.get('response_type')
        results = item.get('results', [])

        passed = sum(r.get('status') == 'Passed' for r in results)
        failed = sum(r.get('status') == 'Failed' for r in results)
        total = passed + failed

        results_per_turn.append({
            'turn_index': turn_index,
            'response_type': response_type,
            'total': total,
            'passed': passed,
            'failed': failed
        })

        if response_type == 'response' and failed > 0:
            resp.append(f'❗ NON FINAL TURN {turn_index} FAILING ON {failed} INSTRUCTIONS ❗')
            task_fail = True

        if total > 0:
            fail_rate = round(failed * 100 / total)
            if response_type == 'nova_response':
                nova_fail = fail_rate
            elif response_type.endswith('_response'):
                frontier_fail_rates.append(fail_rate)

    frontier_fail = round(sum(frontier_fail_rates) / len(frontier_fail_rates)) if frontier_fail_rates else 0

    # Classification logic
    if nova_fail is not None and nova_fail >= 50:
        if frontier_fail >= 80:
            classification = 'EXPERT'
        elif frontier_fail >= 50:
            classification = 'HARD'
        else:
            classification = 'MEDIUM'
    else:
        classification, task_fail = 'N/A', True

    resp.append(f"Nova Fail: {nova_fail}%, Frontier Fail: {frontier_fail}%")
    result = {'task_fail': task_fail, 'text': resp, 'results_per_turn': results_per_turn, 'classification': classification}
    return result
