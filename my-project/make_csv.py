import csv
import os

def load_system_prompt_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        return None
        
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        is_system = False
        system_lines = []
        for line in lines:
            if line.startswith('system: |-'):
                is_system = True
                continue
            if line.startswith('config:'):
                is_system = False
                break
                
            if is_system:
                system_lines.append(line)
        
        if system_lines:
            # We strip trailing newlines but keep the internal ones
            return ''.join(system_lines).rstrip('\n')
            
    except Exception as e:
        print(f"Error reading {yaml_path}: {e}")
        
    return None

def main():
    roles_file = 'data/chatgpt_roles.csv'
    progress_file = 'data/progress.csv'
    output_file = 'data/chatgpt_roles_new.csv'
    
    # Read the original roles
    original_roles = []
    with open(roles_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row: 
                original_roles.append(row[0])
                
    # Read the progress file
    new_prompts = {}
    with open(progress_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['status'] == 'done':
                index = int(row['index'])
                yaml_path = row['output_file'].replace('\\', '/')
                
                system_prompt = load_system_prompt_from_yaml(yaml_path)
                if system_prompt:
                    new_prompts[index] = system_prompt
                    
    # Generate the new roles list
    updated_roles = []
    for i, role in enumerate(original_roles):
        if i in new_prompts:
            updated_roles.append(new_prompts[i])
        else:
            updated_roles.append(role)
            
    # Write the result
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for role in updated_roles:
            writer.writerow([role])
            
    print(f"Successfully processed {len(original_roles)} rows. Updated {len(new_prompts)} prompts. Wrote to {output_file}")

if __name__ == "__main__":
    main()
