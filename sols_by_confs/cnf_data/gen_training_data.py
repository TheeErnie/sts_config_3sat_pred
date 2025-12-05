import sys
import os
sys.path.append(os.path.abspath('..'))

from utils import sts_properties_from_3sat as s3sp
from utils import new_ganak_handler

GANAK_PATH = '/work/luduslab/sts_3sat/ganak/ganak'

def inner_task(args):
    job_offset, data_type, num_files, dir_path, job_breakdown = args

    # lists for number of solutions for later dataframe
    num_sols = []
    num_neg_lits = []
    pasch_counts = []
    mitre_counts = []
    fano_line_counts = []
    grid_counts = []
    prism_counts = []
    hexagon_counts = []
    crown_counts = []
    pasch_counts_r = []
    mitre_counts_r = []
    fano_line_counts_r = []
    grid_counts_r = []
    prism_counts_r = []
    hexagon_counts_r = []
    crown_counts_r = []


    for i in range(int(num_files/job_breakdown)):
        # grab file path for each 3-SAT instance
        file_index = i + job_offset
        sat_file_path = dir_path + f'{data_type}/{data_type}_no{file_index}.cnf'

        # run ganak and save results
        ganak_results = new_ganak_handler.run_ganak_parsed(cnf_path=sat_file_path,ganak_path=GANAK_PATH)
        num_sols.append(ganak_results.model_count_exact)

        # get num negative lits count
        num_neg_lits.append(s3sp.negated_lit_count(file_path=sat_file_path))

        # lookuptable representation of literals with consideration of literals being flipped
        lookup_table = s3sp.sts3sat_lookup_table_from_file(file_path=sat_file_path)

        # count all configuration types
        pasch_counts_r.append(s3sp.count_pasch_configurations_lt(lookup_table))
        mitre_counts_r.append(s3sp.count_mitre_configurations(lookup_table))
        fano_line_counts_r.append(s3sp.count_fano_line_configurations(lookup_table))
        grid_counts_r.append(s3sp.count_grid_configurations(lookup_table))
        prism_counts_r.append(s3sp.count_prism_configurations(lookup_table))
        hexagon_counts_r.append(s3sp.count_hexagon_configurations(lookup_table))
        crown_counts_r.append(s3sp.count_crown_configurations(lookup_table))

        # counting configs based on sts alone
        clauses = s3sp.cnf_to_sts(file_path=sat_file_path)
        lookup_table = s3sp.sts_lookup_table(clauses=clauses)
        pasch_counts.append(s3sp.count_pasch_configurations_lt(lookup_table))
        mitre_counts.append(s3sp.count_mitre_configurations(lookup_table))
        fano_line_counts.append(s3sp.count_fano_line_configurations(lookup_table))
        grid_counts.append(s3sp.count_grid_configurations(lookup_table))
        prism_counts.append(s3sp.count_prism_configurations(lookup_table))
        hexagon_counts.append(s3sp.count_hexagon_configurations(lookup_table))
        crown_counts.append(s3sp.count_crown_configurations(lookup_table))

    result_data = {
        f'sol_count': num_sols,
        f'neg_lits_count': num_neg_lits,
        f'pasch_count': pasch_counts,
        f'mitre_count': mitre_counts,
        f'fano_line_count': fano_line_counts,
        f'grid_count': grid_counts,
        f'prism_count': prism_counts,
        f'hexagon_count': hexagon_counts,
        f'crown_count': crown_counts,
        f'pasch_count_r': pasch_counts_r,
        f'mitre_count_r': mitre_counts_r,
        f'fano_line_count_r': fano_line_counts_r,
        f'grid_count_r': grid_counts_r,
        f'prism_count_r': prism_counts_r,
        f'hexagon_count_r': hexagon_counts_r,
        f'crown_count_r': crown_counts_r,
    }
    return result_data


import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_FILES = 5000000
DATA_TYPE = 'train'
DIR_PATH = '/work/luduslab/sts_3sat/sols_by_confs/cnf_data/'
THREADS = 10
SLURM_ID = int(sys.argv[1]) # range is [0-4]

completed_workers = 0
results_csv_path = DIR_PATH + f'{DATA_TYPE}/{DATA_TYPE}_sub{SLURM_ID}_data.csv'
write_header = True

with ProcessPoolExecutor(max_workers=THREADS) as executor:
    futures = [executor.submit(inner_task, (job_id*100000 + SLURM_ID*1000000, DATA_TYPE, NUM_FILES, DIR_PATH, THREADS*5)) for job_id in range(THREADS)]
    for future in as_completed(futures):
        result_data = future.result()

        df_partial = pd.DataFrame(result_data)

        df_partial.to_csv(
            results_csv_path,
            mode='a',
            header=write_header, 
            index=False
        )

        write_header = False
        completed_workers += 1
        print(f'{completed_workers}/{THREADS} jobs completed')

print(f'Results written to {results_csv_path}')

