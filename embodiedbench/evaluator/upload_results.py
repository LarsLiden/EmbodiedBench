import os
import json
RESULTS_ROOT = "./running/eb_alfred/"
OUTPUT_FILE = "./embodiedbench/combined_results.json"

def combine_results():
    # ~/EmbodiedBench/running/eb_alfred/gpt-4o_baseline/base/results/summary.json
    results = {}
    # Retrieve list of all directories in RESULTS_ROOT
    model_dirs = [d for d in os.listdir(RESULTS_ROOT) if os.path.isdir(os.path.join(RESULTS_ROOT, d))]
    for model_name in model_dirs:
        dir_path = os.path.join(RESULTS_ROOT, model_name)
        print(f"Combining results in {dir_path}")
        test_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        tests = {}
        for test_dir in test_dirs:
            test_path = os.path.join(dir_path, test_dir)
            result_path = os.path.join(test_path, 'results', "summary.json")
            print(f"  Processing {result_path}")
            with open(result_path, 'r') as f:
                data = json.load(f)
                tests[test_dir] = data
            results[model_name] = tests

    with open(OUTPUT_FILE, 'w') as f:
        print(f"Saving combined results to {OUTPUT_FILE}")
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    combine_results()
