
# !/usr/bin/env python3
"""
Publish Compiler for AI Experiment Framework

This script compiles and packages a specific sub-project along with its key experimental results
for publication purposes. It extracts the relevant code and CSV result files from the experiment
directories and creates a standalone project folder suitable for paper publication.
"""

import os
import sys
import shutil
import argparse
import yaml
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_publish_config(config_path):
    """Create a default publish configuration file."""
    default_config = {
        'project_name': '',
        'experiments': [],
        'source_dirs': [],  # New field for multiple source directories
        'output_dir': './published_project',
        'include_configs': True,
        'include_results': True,
        'include_workflow': False,
        'additional_files': []
    }

    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

    print(f"Default publish configuration created at: {config_path}")


def find_csv_files(log_dir, experiment_id):
    """Find all CSV files in the experiment directory."""
    csv_files = []
    exp_path = Path(log_dir) / experiment_id

    if not exp_path.exists():
        # Try alternative path structure (full path from logs root)
        exp_path = Path(log_dir) / experiment_id

    if exp_path.exists():
        for csv_file in exp_path.rglob("*.csv"):
            csv_files.append(csv_file)
    else:
        print(f"Experiment directory not found: {exp_path}")

    return csv_files


def copy_source_directories(sources_dir, dst_dir, source_dirs):
    """Copy multiple source directories."""
    dst_code_dir = Path(dst_dir) / 'sources_root'
    dst_code_dir.mkdir(parents=True, exist_ok=True)

    for src_dir_name in source_dirs:
        src_project_dir = Path(sources_dir) / src_dir_name
        if src_project_dir.exists():
            dst_project_dir = dst_code_dir / src_dir_name
            shutil.copytree(src_project_dir, dst_project_dir)
            print(f"Copied source code from {src_project_dir} to {dst_project_dir}")
        else:
            print(f"Source directory not found: {src_project_dir}")


def copy_config_files(config_dir, dst_dir, project_name):
    """Copy configuration files for the specific project."""
    src_config_dir = Path(config_dir) / project_name
    dst_config_dir = Path(dst_dir) / 'configs' / project_name

    if src_config_dir.exists():
        shutil.copytree(src_config_dir, dst_config_dir)
        print(f"Copied config files from {src_config_dir} to {dst_config_dir}")
    else:
        print(f"Config directory not found: {src_config_dir}")


def copy_workflow_code(workflow_src_dir, dst_dir):
    """Copy workflow code to the published project."""
    dst_workflow_dir = Path(dst_dir) / 'workflow'

    if workflow_src_dir.exists():
        shutil.copytree(workflow_src_dir, dst_workflow_dir)
        print(f"Copied workflow code from {workflow_src_dir} to {dst_workflow_dir}")
    else:
        print(f"Workflow directory not found: {workflow_src_dir}")


def copy_result_files(csv_files, dst_dir):
    """Copy result CSV files to the published project."""
    dst_results_dir = Path(dst_dir) / 'logs'
    dst_results_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        # Preserve relative path structure in results directory
        relative_path = csv_file.relative_to(csv_file.parent.parent.parent.parent)
        dst_file = dst_results_dir / relative_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(csv_file, dst_file)
        print(f"Copied result file: {csv_file} -> {dst_file}")


def create_license_file(project_root, dst_dir, source_dirs):
    """Create a unified LICENSE file from root and source directories."""
    license_content = ""

    # Check for LICENSE file in project root
    root_license = Path(project_root) / "LICENSE"
    if root_license.exists():
        with open(root_license, 'r') as f:
            license_content += f"This project adapts the structure from https://github.com/hujinCN/aiworkflow/ \n"
            license_content += f"==== LICENSE of https://github.com/hujinCN/aiworkflow/ ====\n\n"
            license_content += f.read()
            license_content += "\n\n"
    else:
        print("No LICENSE file found in project root")

    # Check for LICENSE files in source directories
    for src_dir_name in source_dirs:
        src_license = Path(project_root) / "sources_root" / src_dir_name / "LICENSE"
        if src_license.exists():
            with open(src_license, 'r') as f:
                license_content += f"==== {src_dir_name.upper()} LICENSE ====\n\n"
                license_content += f.read()
                license_content += "\n\n"
        else:
            print(f"No LICENSE file found in {src_dir_name}")

    # Write unified license file
    if license_content:
        license_file = Path(dst_dir) / "LICENSE"
        with open(license_file, 'w') as f:
            f.write(license_content)
        print(f"Created unified LICENSE file: {license_file}")
    else:
        print("No license files found to combine")


def create_readme(dst_dir, project_name, experiment_ids, source_dirs, include_workflow):
    """Create a README file for the published project."""
    readme_file = Path(dst_dir) / 'README.md'

    readme_content = f"""# Published Research Project: {project_name}

## Overview

This project contains the code and results for the research experiment '{project_name}'.

## Directory Structure

```
"""

    readme_content += f"{dst_dir.name}/\n"
    readme_content += "|-- sources_root/\n"

    for src_dir in source_dirs:
        readme_content += f"|   |-- {src_dir}/\n"

    if include_workflow:
        readme_content += "|-- workflow/\n"

    readme_content += "|-- configs/\n"
    readme_content += "|-- results/\n"
    readme_content += "|-- README.md\n"
    readme_content += "```\n\n"

    readme_content += """## Quick Start Guide


1. Run the experiment:
   ```bash
   PYTHONPATH=./:./sources_root:./sources_root/cifar_adv_test python sources_root/cifar_adv_test/main.py -c base -i experiment_id -d "your_experiment_description"
   ```

   Or run using the script:
   ```bash
   cd sources_root/cifar_adv_test
   ./run.sh
   ```

## Notes

This README provides only the basic directory structure. Please update with detailed information about the project, how to run experiments, and how to interpret results.

"""

    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"Created README file: {readme_file}")


def main():
    parser = argparse.ArgumentParser(description='Publish Compiler for AI Experiment Framework')
    parser.add_argument('-c', '--config', help='Path to publish configuration YAML file')
    parser.add_argument('--create-config', action='store_true',
                        help='Create a default configuration file')
    parser.add_argument('-p', '--project-root', default='.',
                        help='Project root directory')

    args = parser.parse_args()

    # Handle config creation
    if args.create_config:
        config_path = args.config if args.config else 'publish_config.yaml'
        create_publish_config(config_path)
        return

    # Load configuration
    config_path = args.config if args.config else 'publish_config.yaml'
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Create a default config with: python publish_compiler.py --create-config")
        sys.exit(1)

    config = load_config(config_path)

    # Get paths
    project_root = Path(args.project_root).resolve()
    sources_dir = project_root / 'sources_root'
    configs_dir = project_root / 'configs'
    logs_dir = project_root / 'logs'  # Using local logs dir as per updated path_cfg.yml
    workflow_dir = project_root / 'workflow'

    # Create output directory
    output_dir = Path(config['output_dir']).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Publishing project to: {output_dir}")

    # Copy source directories
    if 'source_dirs' in config and config['source_dirs']:
        copy_source_directories(sources_dir, output_dir, config['source_dirs'])

    # Copy workflow code if requested
    if config.get('include_workflow', False):
        copy_workflow_code(workflow_dir, output_dir)

    # Copy config files
    if config['include_configs']:
        copy_config_files(configs_dir, output_dir, config['project_name'])

    # Copy result files
    if config['include_results']:
        all_csv_files = []
        for exp_id in config['experiments']:
            csv_files = find_csv_files(logs_dir, exp_id)
            all_csv_files.extend(csv_files)

        if all_csv_files:
            copy_result_files(all_csv_files, output_dir)
        else:
            print("No CSV result files found for the specified experiment IDs")

    # Copy additional files
    for file_path in config['additional_files']:
        src_file = Path(file_path)
        if src_file.exists():
            dst_file = output_dir / src_file.name
            shutil.copy(src_file, dst_file)
            print(f"Copied additional file: {src_file} -> {dst_file}")

    # Create unified LICENSE file
    create_license_file(project_root, output_dir, config.get('source_dirs', []))

    # Create README
    create_readme(output_dir, config['project_name'], config['experiments'],
                  config.get('source_dirs', []), config.get('include_workflow', False))

    print(f"\nPublication completed successfully!")
    print(f"Published project is located at: {output_dir}")


if __name__ == '__main__':
    main()