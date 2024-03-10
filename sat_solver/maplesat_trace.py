import subprocess
import sys
import os

def run_maplesat(cnf_formula, output_file, trace_file, maplesat_path):
    with open('temp_input.cnf', 'w') as temp_input:
        temp_input.write(cnf_formula)
    process = subprocess.run([maplesat_path, 'temp_input.cnf', output_file, f'-trace-out={trace_file}'],
                             stdout=subprocess.PIPE, text=True)
    os.remove('temp_input.cnf')
    return process.stdout.strip().split('\n')[-1]

def process_maplesat_output(output_file, sat_status):
    with open(output_file, 'r') as file:
        content = file.read().strip()
    if sat_status == 'UNSATISFIABLE':
        return 'UNSAT', process_drup(content)
    else:
        return 'SAT', content

def process_drup(drup_output):
    return ' '.join([line for line in drup_output.split('\n') if not line.startswith('d ')])

def main(input_file, output_file, maplesat_path):
    with open(input_file, 'r') as in_file, open(output_file, 'w') as out_file:
        for cnf_formula in in_file:
            trace_file = 'temp_trace.trc'
            result_file = 'temp_result.txt'
            
            sat_status = run_maplesat(cnf_formula, result_file, trace_file, maplesat_path)
            
            status, result = process_maplesat_output(result_file, sat_status)
            with open(trace_file, 'r') as trc:
                trace = trc.read().strip()
            
            out_file.write(f'{cnf_formula.strip()} [SEP] {trace} {status}\n')
            
            os.remove(trace_file)
            os.remove(result_file)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python maplesat_trace.py INPUT_FILE OUTPUT_FILE MAPLESAT_STATIC_PATH")
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
