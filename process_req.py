import re

def process_requirements_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding="utf-16le") as file:
            lines = file.readlines()

        # Use a regular expression to replace one or more whitespace characters with '=='
        processed_lines = [re.sub(r'\s+', '==', line.strip()) + '\n' for line in lines if line.strip()]

        with open(output_file, 'w', encoding="utf-8") as file:
            file.writelines(processed_lines)
        
        print(f"Processed requirements have been saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    input_file = 'requirements.txt'
    output_file = 'processed_requirements.txt'
    process_requirements_file(input_file, output_file)
