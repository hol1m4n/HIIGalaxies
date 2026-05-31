from astropy.table import Table
import pandas as pd
import nested_sampler as ns 

LSdata_df = pd.read_csv('Compilation2026.csv',comment='#',index_col=False, dtype={'GEHR_id': str})
LS_tab = Table.from_pandas(LSdata_df)

z_cut = LS_tab[(LS_tab['origin_id'] == 0.0) | ((LS_tab['origin_id'] != 0.0) & (LS_tab['z_or_mu'] <= 0.10))]


ns.Lsig_Ho_sampler(
    data_frame = z_cut,
    distance_estimator_set = 'sm_r.csv',
    estimator_error_kind = 'sigma_w',
    main_title = r'L-sigma test: Local Universe (z <= 0.10), Set: Cep-nZeR error: $\sigma_w$',
    folder_name = 'LU_smr_sigma_w',
    analysis_mode = 'Low',
    id_prefix = 'low_smr_sw'
)






LSdata_df = pd.read_csv('Compilation2026.csv',comment='#',index_col=False, dtype={'GEHR_id': str})
LS_tab = Table.from_pandas(LSdata_df)

z_cut = LS_tab[(LS_tab['origin_id'] == 0.0) | ((LS_tab['origin_id'] != 0.0) & (LS_tab['z_or_mu'] <= 1.0))]


ns.Lsig_Ho_sampler(
    data_frame = z_cut,
    distance_estimator_set = 'sm_r.csv',
    estimator_error_kind = 'sigma_w',
    main_title = r'L-sigma test: Moderate redshift (z <= 1.0), Set: Cep-nZeR error: $\sigma_w$',
    folder_name = 'MU_smr_sigma_w',
    analysis_mode = 'Moderate',
    id_prefix = 'mod_smr_sw'
)





LSdata_df = pd.read_csv('Compilation2026.csv',comment='#',index_col=False, dtype={'GEHR_id': str})
LS_tab = Table.from_pandas(LSdata_df)

z_cut = LS_tab[(LS_tab['origin_id'] == 0.0) | ((LS_tab['origin_id'] != 0.0) & (LS_tab['z_or_mu'] <= 1000))]


ns.Lsig_Ho_sampler(
    data_frame = z_cut,
    distance_estimator_set = 'sm_r.csv',
    estimator_error_kind = 'sigma_w',
    main_title = r'L-sigma test: Full cosmology (z $\sim \infty$), Set: Cep-nZeR error: $\sigma_w$',
    folder_name = 'HU_smr_sigma_w',
    analysis_mode = 'High',
    id_prefix = 'high_smr_sw'
)



'''

LSdata_df = pd.read_csv('Compilation2026.csv',comment='#',index_col=False, dtype={'GEHR_id': str})
LS_tab = Table.from_pandas(LSdata_df)

z_cut = LS_tab[(LS_tab['origin_id'] == 0.0) | ((LS_tab['origin_id'] != 0.0) & (LS_tab['z_or_mu'] <= 0.10))]


ns.Lsig_Ho_sampler(
    data_frame = z_cut,
    distance_estimator_set = 't_r.csv',
    estimator_error_kind = 'sigma_w',
    main_title = r'L-sigma test: Local Universe (z <= 0.10), Set: TRGB error: $\sigma_w$',
    folder_name = 'LU_tr_sigma_w',
    analysis_mode = 'Low',
    id_prefix = 'low_tr_sw'
)






LSdata_df = pd.read_csv('Compilation2026.csv',comment='#',index_col=False, dtype={'GEHR_id': str})
LS_tab = Table.from_pandas(LSdata_df)

z_cut = LS_tab[(LS_tab['origin_id'] == 0.0) | ((LS_tab['origin_id'] != 0.0) & (LS_tab['z_or_mu'] <= 1.0))]


ns.Lsig_Ho_sampler(
    data_frame = z_cut,
    distance_estimator_set = 't_r.csv',
    estimator_error_kind = 'sigma_w',
    main_title = r'L-sigma test: Moderate redshift (z <= 1.0), Set: TRGB error: $\sigma_w$',
    folder_name = 'MU_tr_sigma_w',
    analysis_mode = 'Moderate',
    id_prefix = 'mod_tr_sw'
)





LSdata_df = pd.read_csv('Compilation2026.csv',comment='#',index_col=False, dtype={'GEHR_id': str})
LS_tab = Table.from_pandas(LSdata_df)

z_cut = LS_tab[(LS_tab['origin_id'] == 0.0) | ((LS_tab['origin_id'] != 0.0) & (LS_tab['z_or_mu'] <= 1000))]


ns.Lsig_Ho_sampler(
    data_frame = z_cut,
    distance_estimator_set = 't_r.csv',
    estimator_error_kind = 'sigma_w',
    main_title = r'L-sigma test: Full cosmology (z $\sim \infty$), Set: TRGB error: $\sigma_w$',
    folder_name = 'HU_tr_sigma_w',
    analysis_mode = 'High',
    id_prefix = 'high_tr_sw'
)


'''









