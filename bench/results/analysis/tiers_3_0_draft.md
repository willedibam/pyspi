# Draft cost tiers for pyspi 3.0

Draft, from a partial cost grid at pyspi 3.0.0 (ce4b20e): white noise, one core, M in {4, 8, 16, 32} and T in {250, ..., 4000}, some large cells still missing.
Not yet applied to `benchmarked_p*.yaml`.

Draft tier = cheapest-first cut at M=16, T=1000, one core (counts 184/207/219/228/231).
Config pN = every row whose tier is pN or cheaper.
Cost = amortised seconds per dataset (cache-sharing SPIs carry equal shares).
Exponents from log cost = a + p_M log M + q_T log T over the finished cells; `–` = no fit (cost < 0.05 s in most cells), projection then uses (M/16)²(T/1000).
`rmse` ≥ 0.4 marks an unreliable fit (wavelets, dcorr, bary_dtw_corr, mgcx, psi).
Machine-readable copy: `tiers_3_0_draft.csv`.
`shipped` is the smallest current `benchmarked_p*` config holding the SPI.

| tier | n added | cumulative n | cumulative s at M16 T1000 |
|---|---:|---:|---:|
| p80 | 184 | 184 | 33 |
| p90 | 23 | 207 | 95 |
| p95 | 12 | 219 | 431 |
| p99 | 9 | 228 | 2,548 |
| full | 3 | 231 | 5,116 |

## Enters at p80 (184 SPIs)

| SPI | family | module | s M16 T1000 | p_M | q_T | rmse | proj s M16 T4000 | shipped |
|---|---|---|---:|---:|---:|---:|---:|---|
| `cce_gaussian_n-5` | cce | infotheory | 0.366 | 1.98 | 0.38 | 0.10 | 0.677 | p80 |
| `cddtf_p-bic_mean_fmin-0-25_fmax-0-5` | cddtf | var | 0.0111 | – | – | – | 0.0444 | none |
| `cddtf_p-bic_mean_fmin-0_fmax-0-25` | cddtf | var | 0.0111 | – | – | – | 0.0444 | none |
| `cddtf_p-bic_mean_fmin-0_fmax-0-5` | cddtf | var | 0.0111 | – | – | – | 0.0444 | none |
| `ce_kozachenko` | ce | infotheory | 0.0121 | – | – | – | 0.0482 | p80 |
| `ce_kernel_W-0.5` | ce | infotheory | 0.0281 | – | – | – | 0.112 | p80 |
| `cgc_p-bic` | cgc | var | 0.0111 | – | – | – | 0.0444 | none |
| `cgc_p-1` | cgc | var | 0.0164 | – | – | – | 0.0657 | none |
| `cgc_p-2` | cgc | var | 0.162 | – | – | – | 0.648 | none |
| `cgpdc_p-bic_mean_fmin-0-25_fmax-0-5` | cgpdc | var | 0.0111 | – | – | – | 0.0444 | none |
| `cgpdc_p-bic_mean_fmin-0_fmax-0-25` | cgpdc | var | 0.0111 | – | – | – | 0.0444 | none |
| `cgpdc_p-bic_mean_fmin-0_fmax-0-5` | cgpdc | var | 0.0111 | – | – | – | 0.0444 | none |
| `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5` | cohmag | spectral | 0.00622 | – | – | – | 0.0249 | p80 |
| `cohmag_multitaper_max_fs-1_fmin-0_fmax-0-5` | cohmag | spectral | 0.00622 | – | – | – | 0.0249 | p80 |
| `cohmag_multitaper_max_fs-1_fmin-0_fmax-0-25` | cohmag | spectral | 0.00622 | – | – | – | 0.0249 | p80 |
| `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | cohmag | spectral | 0.00622 | – | – | – | 0.0249 | p80 |
| `cohmag_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | cohmag | spectral | 0.00622 | – | – | – | 0.0249 | p80 |
| `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` | cohmag | spectral | 0.00622 | – | – | – | 0.0249 | p80 |
| `coint_johansen_max_eig_stat_order-0_ardiff-1` | coint | misc | 0.0925 | 2.01 | 0.10 | 0.01 | 0.107 | p80 |
| `coint_johansen_trace_stat_order-0_ardiff-1` | coint | misc | 0.0925 | 2.01 | 0.10 | 0.01 | 0.107 | p80 |
| `coint_johansen_max_eig_stat_order-1_ardiff-1` | coint | misc | 0.101 | 2.02 | 0.15 | 0.01 | 0.125 | p80 |
| `coint_johansen_trace_stat_order-1_ardiff-1` | coint | misc | 0.101 | 2.02 | 0.15 | 0.01 | 0.125 | p80 |
| `coint_johansen_max_eig_stat_order-1_ardiff-10` | coint | misc | 0.177 | 1.99 | 0.40 | 0.08 | 0.328 | p80 |
| `coint_johansen_trace_stat_order-1_ardiff-10` | coint | misc | 0.177 | 1.99 | 0.40 | 0.08 | 0.328 | p80 |
| `coint_johansen_trace_stat_order-0_ardiff-10` | coint | misc | 0.385 | 1.12 | 0.26 | 0.50 | 0.4 | p80 |
| `coint_johansen_max_eig_stat_order-0_ardiff-10` | coint | misc | 0.385 | 1.12 | 0.26 | 0.50 | 0.4 | p80 |
| `cov_EmpiricalCovariance` | cov | basic | 0.0218 | – | – | – | 0.0874 | p80 |
| `cov_MinCovDet` | cov | basic | 0.35 | 0.21 | 0.45 | 0.08 | 0.699 | p80 |
| `cpdc_p-bic_mean_fmin-0-25_fmax-0-5` | cpdc | var | 0.0111 | – | – | – | 0.0444 | none |
| `cpdc_p-bic_mean_fmin-0_fmax-0-25` | cpdc | var | 0.0111 | – | – | – | 0.0444 | none |
| `cpdc_p-bic_mean_fmin-0_fmax-0-5` | cpdc | var | 0.0111 | – | – | – | 0.0444 | none |
| `dcoh_multitaper_max_fs-1_fmin-0_fmax-0-25` | dcoh | spectral | 1.01 | 2.09 | 0.71 | 0.07 | 2.77 | p80 |
| `dcoh_multitaper_mean_fs-1_fmin-0_fmax-0-5` | dcoh | spectral | 1.01 | 2.09 | 0.71 | 0.07 | 2.77 | p80 |
| `dcoh_multitaper_mean_fs-1_fmin-0_fmax-0-25` | dcoh | spectral | 1.01 | 2.09 | 0.71 | 0.07 | 2.77 | p80 |
| `dcoh_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | dcoh | spectral | 1.01 | 2.09 | 0.71 | 0.07 | 2.77 | p80 |
| `dcoh_multitaper_max_fs-1_fmin-0_fmax-0-5` | dcoh | spectral | 1.01 | 2.09 | 0.71 | 0.07 | 2.77 | p80 |
| `dcoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | dcoh | spectral | 1.01 | 2.09 | 0.71 | 0.07 | 2.77 | p80 |
| `dcorr` | dcorr | distance | 0.439 | 1.42 | 2.23 | 0.87 | 20.7 | p90 |
| `dcorr_biased` | dcorr | distance | 0.829 | 1.68 | 2.68 | 0.30 | 39.9 | p90 |
| `di_gaussian_n-5` | di | infotheory | 0.59 | 2.03 | 0.31 | 0.07 | 0.963 | p80 |
| `dspli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | dspli | spectral | 0.0105 | – | – | – | 0.042 | p80 |
| `dspli_multitaper_mean_fs-1_fmin-0_fmax-0-25` | dspli | spectral | 0.0105 | – | – | – | 0.042 | p80 |
| `dspli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | dspli | spectral | 0.0105 | – | – | – | 0.042 | p80 |
| `dswpli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | dswpli | spectral | 0.0173 | – | – | – | 0.0694 | p80 |
| `dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | dswpli | spectral | 0.0173 | – | – | – | 0.0694 | p80 |
| `dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-25` | dswpli | spectral | 0.0173 | – | – | – | 0.0694 | p80 |
| `dtw_constraint-sakoe-chiba_radius-auto` | dtw | distance | 0.0616 | 2.05 | 1.44 | 0.06 | 0.422 | p80 |
| `dtw_constraint-itakura` | dtw | distance | 0.585 | 0.84 | 0.69 | 0.74 | 1.54 | p80 |
| `dtw` | dtw | distance | 0.929 | 2.09 | 2.01 | 0.36 | 12.8 | p80 |
| `edge_excessvar` | edge | misc | 0.000741 | – | – | – | 0.00296 | none |
| `edge_acf1` | edge | misc | 0.00126 | – | – | – | 0.00506 | none |
| `edge_topshare_f-0-05` | edge | misc | 0.807 | 0.76 | 0.77 | 0.38 | 1.62 | none |
| `es_asym_q-0-9` | es | events | 0.00662 | – | – | – | 0.0265 | none |
| `es_strength_q-0-9` | es | events | 0.00662 | – | – | – | 0.0265 | none |
| `es_strength_q-0-95` | es | events | 0.0105 | – | – | – | 0.0421 | none |
| `es_asym_q-0-95` | es | events | 0.0105 | – | – | – | 0.0421 | none |
| `gc_gaussian_k-1_kt-1_l-1_lt-1` | gc | infotheory | 0.0889 | 2.10 | 0.23 | 0.04 | 0.13 | p80 |
| `gc_gaussian_MAX-CORR-AIS_k-max-10_tau-max-2` | gc | infotheory | 0.268 | 1.63 | 0.45 | 0.16 | 0.605 | p80 |
| `hsic_biased` | hsic | distance | 1.03 | 1.57 | 2.63 | 0.38 | 65.2 | p95 |
| `hsic` | hsic | distance | 1.05 | 1.35 | 2.46 | 0.32 | 46.5 | p95 |
| `icoh_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | icoh | spectral | 0.00636 | – | – | – | 0.0255 | p80 |
| `icoh_multitaper_mean_fs-1_fmin-0_fmax-0-5` | icoh | spectral | 0.00636 | – | – | – | 0.0255 | p80 |
| `icoh_multitaper_mean_fs-1_fmin-0_fmax-0-25` | icoh | spectral | 0.00636 | – | – | – | 0.0255 | p80 |
| `icoh_multitaper_max_fs-1_fmin-0_fmax-0-5` | icoh | spectral | 0.00636 | – | – | – | 0.0255 | p80 |
| `icoh_multitaper_max_fs-1_fmin-0_fmax-0-25` | icoh | spectral | 0.00636 | – | – | – | 0.0255 | p80 |
| `icoh_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | icoh | spectral | 0.00636 | – | – | – | 0.0255 | p80 |
| `ids` | ids | misc | 0.0238 | – | – | – | 0.0953 | p80 |
| `je_kozachenko` | je | infotheory | 0.206 | 1.11 | 0.70 | 0.44 | 0.487 | p80 |
| `je_kernel_W-0.5` | je | infotheory | 0.54 | 2.07 | 1.36 | 0.09 | 3.79 | p80 |
| `lagcorr_onesided_tau-2_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_onesided_tau-1_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_lag_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_peak_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_asym_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_onesidedpeak_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_peak_L-10` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_lag_L-10` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_onesided_tau-3_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_onesided_tau-5_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_onesided_tau-10_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_asym_L-10` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `lagcorr_onesided_tau-1e_L-None` | lagcorr | basic | 0.00452 | – | – | – | 0.0181 | p80 |
| `liang_rate_k-1` | liang | causal | 0.0045 | – | – | – | 0.018 | none |
| `liang_norm_k-1` | liang | causal | 0.0045 | – | – | – | 0.018 | none |
| `liang-cond_rate_k-1` | liang-cond | causal | 0.0045 | – | – | – | 0.018 | none |
| `liang-cond_norm_k-1` | liang-cond | causal | 0.0045 | – | – | – | 0.018 | none |
| `mi_gaussian` | mi | infotheory | 0.000396 | – | – | – | 0.00158 | p80 |
| `mi_kraskov_NN-4` | mi | infotheory | 0.559 | 2.08 | 0.92 | 0.05 | 2.07 | p80 |
| `mi_kernel_W-0.25` | mi | infotheory | 0.724 | 2.08 | 1.18 | 0.05 | 3.83 | p80 |
| `mi_kraskov_NN-4_DCE-AUTO` | mi | infotheory | 0.736 | 1.98 | 0.98 | 0.24 | 2.92 | p80 |
| `mi_kernel_W-0.25_DCE-AUTO` | mi | infotheory | 0.747 | 2.08 | 1.15 | 0.06 | 3.82 | p80 |
| `nli_L_m-5_tau-1_k-5` | nli | causal | 0.293 | 1.29 | 1.44 | 0.15 | 2.56 | none |
| `nli_N_m-5_tau-1_k-5` | nli | causal | 0.293 | 1.29 | 1.44 | 0.15 | 2.56 | none |
| `nli_H_m-5_tau-1_k-5` | nli | causal | 0.293 | 1.29 | 1.44 | 0.15 | 2.56 | none |
| `nli_L_m-3_tau-1_k-5` | nli | causal | 0.583 | 1.26 | 1.48 | 0.14 | 5.3 | none |
| `pac_mi_excess_p-0-05-0-1_a-0-2-0-45` | pac | crossfreq | 0.0686 | 1.66 | 0.62 | 0.08 | 0.181 | none |
| `pac_mi_excess_p-0-01-0-05_a-0-15-0-4` | pac | crossfreq | 0.0728 | 1.18 | 0.78 | 0.38 | 0.422 | none |
| `pcohmag_p-bic_mean_fmin-0-25_fmax-0-5` | pcohmag | var | 0.0111 | – | – | – | 0.0444 | none |
| `pcohmag_p-bic_mean_fmin-0_fmax-0-25` | pcohmag | var | 0.0111 | – | – | – | 0.0444 | none |
| `pcohmag_p-bic_mean_fmin-0_fmax-0-5` | pcohmag | var | 0.0111 | – | – | – | 0.0444 | none |
| `pcorr_OAS` | pcorr | basic | 0.00101 | – | – | – | 0.00405 | p80 |
| `pcorr_ShrunkCovariance` | pcorr | basic | 0.00124 | – | – | – | 0.00495 | p80 |
| `pcorr_LedoitWolf` | pcorr | basic | 0.0013 | – | – | – | 0.0052 | p80 |
| `pcorr_GraphicalLasso` | pcorr | basic | 0.00489 | – | – | – | 0.0196 | p80 |
| `pcorr_EmpiricalCovariance` | pcorr | basic | 0.0218 | – | – | – | 0.0874 | p80 |
| `pcorr_MinCovDet` | pcorr | basic | 0.35 | 0.21 | 0.45 | 0.08 | 0.699 | p80 |
| `pdist_braycurtis` | pdist | distance | 0.000487 | – | – | – | 0.00195 | p80 |
| `pdist_canberra` | pdist | distance | 0.000603 | – | – | – | 0.00241 | p80 |
| `pdist_chebyshev` | pdist | distance | 0.000736 | – | – | – | 0.00294 | p80 |
| `pdist_cityblock` | pdist | distance | 0.00105 | – | – | – | 0.00422 | p80 |
| `pec_orth_abs` | pec | misc | 0.00504 | – | – | – | 0.0202 | p80 |
| `pec_log` | pec | misc | 0.00596 | – | – | – | 0.0239 | p80 |
| `pec_orth_log_abs` | pec | misc | 0.00601 | – | – | – | 0.024 | p80 |
| `pec_orth_log` | pec | misc | 0.00868 | – | – | – | 0.0347 | p80 |
| `pec_orth` | pec | misc | 0.0106 | – | – | – | 0.0423 | p80 |
| `pec` | pec | misc | 0.978 | -1.75 | -1.16 | 0.62 | 0.0616 | p80 |
| `phase_multitaper_mean_fs-1_fmin-0_fmax-0-5` | phase | spectral | 0.135 | 1.35 | 1.54 | 0.42 | 0.85 | p80 |
| `phase_multitaper_mean_fs-1_fmin-0_fmax-0-25` | phase | spectral | 0.135 | 1.35 | 1.54 | 0.42 | 0.85 | p80 |
| `phase_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | phase | spectral | 0.135 | 1.35 | 1.54 | 0.42 | 0.85 | p80 |
| `phasealigned_welch_nperseg-auto` | phasealigned | spectral | 0.00921 | – | – | – | 0.0369 | none |
| `phi_star_t-1_norm-0` | phi | infotheory | 0.478 | 1.44 | 0.84 | 0.65 | 1.47 | p80 |
| `phi_Geo_t-1_norm-0` | phi | infotheory | 0.508 | 2.07 | 0.09 | 0.12 | 0.654 | p80 |
| `pli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | pli | spectral | 0.0107 | – | – | – | 0.0428 | p80 |
| `pli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | pli | spectral | 0.0107 | – | – | – | 0.0428 | p80 |
| `pli_multitaper_mean_fs-1_fmin-0_fmax-0-25` | pli | spectral | 0.0107 | – | – | – | 0.0428 | p80 |
| `plv_multitaper_max_fs-1_fmin-0-25_fmax-0-5` | plv | spectral | 0.00778 | – | – | – | 0.0311 | p80 |
| `plv_multitaper_max_fs-1_fmin-0_fmax-0-25` | plv | spectral | 0.00778 | – | – | – | 0.0311 | p80 |
| `plv_multitaper_max_fs-1_fmin-0_fmax-0-5` | plv | spectral | 0.00778 | – | – | – | 0.0311 | p80 |
| `plv_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | plv | spectral | 0.00778 | – | – | – | 0.0311 | p80 |
| `plv_multitaper_mean_fs-1_fmin-0_fmax-0-25` | plv | spectral | 0.00778 | – | – | – | 0.0311 | p80 |
| `plv_multitaper_mean_fs-1_fmin-0_fmax-0-5` | plv | spectral | 0.00778 | – | – | – | 0.0311 | p80 |
| `ppc_hilbert_adaptive` | ppc | spectral | 0.199 | 1.40 | 0.90 | 0.25 | 1.02 | none |
| `psi_multitaper_mean_fs-1_fmin-0_fmax-0-25` | psi | spectral | 0.0326 | – | – | – | 0.13 | p80 |
| `psi_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | psi | spectral | 0.033 | – | – | – | 0.132 | p80 |
| `psi_multitaper_mean_fs-1_fmin-0_fmax-0-5` | psi | spectral | 0.0334 | – | – | – | 0.134 | p80 |
| `psi_wavelet_mean_fs-1_fmin-0-25_fmax-0-5_mean` | psi | wavelet | 0.0985 | 2.35 | 1.61 | 0.97 | 1.51 | p80 |
| `psi_wavelet_mean_fs-1_fmin-0_fmax-0-25_mean` | psi | wavelet | 0.106 | 0.31 | 0.38 | 1.34 | 0.571 | p99 |
| `psi_wavelet_mean_fs-1_fmin-0_fmax-0-5_mean` | psi | wavelet | 1.09 | 0.62 | 0.64 | 1.02 | 2.66 | p95 |
| `reci` | reci | causal | 0.649 | 2.03 | 0.25 | 0.17 | 1.07 | p80 |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-None` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_max_fs-1_fmin-0-25_fmax-0-5_order-20` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-25_order-20` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-20` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_max_fs-1_fmin-0-25_fmax-0-5_order-1` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-20` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-20` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-20` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-1` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-1` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-1` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_max_fs-1_fmin-0-25_fmax-0-5_order-None` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-None` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-None` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_nonparametric_max_fs-1_fmin-0-25_fmax-0-5` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_nonparametric_max_fs-1_fmin-1e-05_fmax-0-25` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_nonparametric_max_fs-1_fmin-1e-05_fmax-0-5` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_nonparametric_mean_fs-1_fmin-0-25_fmax-0-5` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_nonparametric_mean_fs-1_fmin-1e-05_fmax-0-25` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_nonparametric_mean_fs-1_fmin-1e-05_fmax-0-5` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-5_order-1` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-25_order-1` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-None` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `sgc_parametric_max_fs-1_fmin-1e-05_fmax-0-25_order-None` | sgc | spectral | 0.149 | 1.92 | 0.61 | 0.10 | 0.375 | p80 |
| `si_gaussian_k-1` | si | infotheory | 0.0372 | – | – | – | 0.149 | p80 |
| `si_kozachenko_k-1` | si | infotheory | 0.289 | 1.92 | 0.95 | 0.03 | 1.11 | p80 |
| `softdtw_constraint-sakoe-chiba` | softdtw | distance | 0.784 | 2.11 | 1.23 | 0.44 | 3.93 | p95 |
| `spearmanr` | spearmanr | basic | 0.00336 | – | – | – | 0.0134 | p80 |
| `tailasym_q-0-95` | tailasym | events | 0.00208 | – | – | – | 0.00832 | none |
| `tailasym_q-0-9` | tailasym | events | 0.00209 | – | – | – | 0.00838 | none |
| `tlmi_gaussian` | tlmi | infotheory | 0.000558 | – | – | – | 0.00223 | p80 |
| `wpli_multitaper_mean_fs-1_fmin-0_fmax-0-25` | wpli | spectral | 0.0135 | – | – | – | 0.0539 | p80 |
| `wpli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | wpli | spectral | 0.0135 | – | – | – | 0.0539 | p80 |
| `wpli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | wpli | spectral | 0.0135 | – | – | – | 0.0539 | p80 |
| `wtc_wavelet_mean_fs-1_fmin-0_fmax-0-5` | wtc | wavelet | 0.298 | 1.58 | 0.22 | 1.37 | 0.893 | none |
| `wtc_wavelet_mean_fs-1_fmin-0_fmax-0-25` | wtc | wavelet | 0.298 | 1.58 | 0.22 | 1.37 | 0.893 | none |
| `wtc_wavelet_mean_fs-1_fmin-0-25_fmax-0-5` | wtc | wavelet | 0.298 | 1.58 | 0.22 | 1.37 | 0.893 | none |
| `xi_tau-0` | xi | basic | 0.00821 | – | – | – | 0.0328 | p80 |
| `xi_tau-1` | xi | basic | 0.00823 | – | – | – | 0.0329 | p80 |
| `xme_gaussian_k1` | xme | infotheory | 0.0331 | – | – | – | 0.133 | p80 |
| `xme_gaussian_k10` | xme | infotheory | 0.0789 | 1.97 | 0.34 | 0.04 | 0.133 | p80 |
| `xme_kozachenko_k1` | xme | infotheory | 0.222 | 1.98 | 0.88 | 0.03 | 0.772 | p80 |
| `xme_kernel_W-0.5_k1` | xme | infotheory | 1.11 | 2.06 | 1.36 | 0.06 | 7.64 | p80 |
| `xpdist_euclidean_tau-1_min_rmse` | xpdist | distance | 0.00436 | – | – | – | 0.0174 | p80 |
| `xpdist_euclidean_tau-1_mean_rmse` | xpdist | distance | 0.00488 | – | – | – | 0.0195 | p80 |

## Enters at p90 (23 SPIs)

| SPI | family | module | s M16 T1000 | p_M | q_T | rmse | proj s M16 T4000 | shipped |
|---|---|---|---:|---:|---:|---:|---:|---|
| `anm` | anm | causal | 2.59 | 1.92 | 1.53 | 0.48 | 36.8 | p95 |
| `cce_kozachenko_n-5` | cce | infotheory | 7.54 | 2.11 | 1.41 | 0.03 | 52.8 | p90 |
| `cds` | cds | causal | 6.96 | 2.12 | 0.45 | 0.07 | 13.6 | p90 |
| `coint_aeg_tstat_trend-ct_autolag-bic_maxlag-10` | coint | misc | 1.96 | 2.10 | 0.43 | 0.10 | 3.71 | p80 |
| `coint_aeg_tstat_trend-ct_autolag-aic_maxlag-10` | coint | misc | 1.99 | 2.08 | 0.48 | 0.14 | 4.06 | p80 |
| `coint_aeg_tstat_trend-c_autolag-aic_maxlag-10` | coint | misc | 2.08 | 2.11 | 0.45 | 0.10 | 3.9 | p80 |
| `dcorrx_maxlag-1` | dcorrx | distance | 2.61 | 1.72 | 2.65 | 0.27 | 151 | p95 |
| `lcss` | lcss | distance | 1.75 | 2.18 | 1.71 | 0.14 | 20 | p90 |
| `lcss_constraint-sakoe-chiba` | lcss | distance | 2 | 2.41 | 1.94 | 0.55 | 39.9 | p90 |
| `lcss_constraint-itakura` | lcss | distance | 2.09 | 2.38 | 1.94 | 0.55 | 43.3 | p90 |
| `si_kernel_W-0.5_k-1` | si | infotheory | 1.46 | 1.97 | 1.40 | 0.05 | 10.5 | p90 |
| `softdtw_constraint-itakura` | softdtw | distance | 3.97 | 2.10 | 1.76 | 0.19 | 45.8 | p95 |
| `te_symbolic_k-3` | te | infotheory | 2.29 | 2.13 | 0.96 | 0.04 | 8.72 | p80 |
| `te_kraskov_NN-4_k-1_kt-1_l-1_lt-1` | te | infotheory | 2.51 | 2.12 | 1.06 | 0.04 | 11.1 | p90 |
| `te_kernel_W-0.25_k-1` | te | infotheory | 2.61 | 2.11 | 1.26 | 0.06 | 15.4 | p90 |
| `te_kernel_W-0.25_DCE-AUTO_k-1` | te | infotheory | 2.7 | 2.11 | 1.24 | 0.06 | 15.5 | p90 |
| `te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1` | te | infotheory | 2.77 | 2.12 | 1.05 | 0.05 | 12 | p90 |
| `te_kraskov_NN-4_DCE-AUTO_k-2_kt-1_l-1_lt-1` | te | infotheory | 4.79 | 2.13 | 1.16 | 0.04 | 24 | p90 |
| `tlmi_kraskov_NN-4` | tlmi | infotheory | 1.12 | 2.09 | 0.91 | 0.05 | 4.1 | p90 |
| `tlmi_kraskov_NN-4_DCE-AUTO` | tlmi | infotheory | 1.29 | 2.09 | 0.89 | 0.06 | 4.57 | p90 |
| `tlmi_kernel_W-0.25` | tlmi | infotheory | 1.45 | 2.09 | 1.17 | 0.06 | 7.58 | p80 |
| `tlmi_kernel_W-0.25_DCE-AUTO` | tlmi | infotheory | 1.48 | 2.09 | 1.15 | 0.06 | 7.62 | p80 |
| `xme_kozachenko_k10` | xme | infotheory | 1.99 | 2.01 | 1.64 | 0.03 | 18.9 | p90 |

## Enters at p95 (12 SPIs)

| SPI | family | module | s M16 T1000 | p_M | q_T | rmse | proj s M16 T4000 | shipped |
|---|---|---|---:|---:|---:|---:|---:|---|
| `bary_dtw_corr` | bary | distance | 13.7 | 2.31 | 2.36 | 0.41 | 446 | p80 |
| `ccm_E-1_convergence` | ccm | causal | 20.6 | 1.45 | 0.61 | 0.12 | 57.6 | none |
| `ccm_E-1_max` | ccm | causal | 20.6 | 1.45 | 0.61 | 0.12 | 57.6 | p99 |
| `ccm_E-1_mean` | ccm | causal | 20.6 | 1.45 | 0.61 | 0.12 | 57.6 | p99 |
| `dcorrx_maxlag-10` | dcorrx | distance | 15.8 | 1.47 | 2.40 | 0.22 | 526 | p99 |
| `di_kraskov_NN-4_n-5` | di | infotheory | 46 | 2.13 | 1.49 | 0.03 | 355 | none |
| `lccm_E-None_k-5_tp` | lccm | causal | 39.2 | 1.76 | 0.71 | 0.17 | 108 | none |
| `lccm_E-None_k-5_skill` | lccm | causal | 39.2 | 1.76 | 0.71 | 0.17 | 108 | none |
| `mgc` | mgc | distance | 34.6 | 2.11 | 2.23 | 0.48 | 1,028 | p95 |
| `softdtw` | softdtw | distance | 9.11 | 2.14 | 1.90 | 0.05 | 128 | p95 |
| `te_kraskov_NN-4_MAX-CORR-AIS_k-max-10_tau-max-4` | te | infotheory | 28.6 | 1.47 | 1.52 | 0.12 | 286 | p95 |
| `te_kraskov_NN-4_DCE-AUTO_MAX-CORR-AIS_k-max-10_tau-max-4` | te | infotheory | 48.2 | 1.37 | 1.49 | 0.09 | 425 | p95 |

## Enters at p99 (9 SPIs)

| SPI | family | module | s M16 T1000 | p_M | q_T | rmse | proj s M16 T4000 | shipped |
|---|---|---|---:|---:|---:|---:|---:|---|
| `bary_softdtw_corr` | bary | distance | 419 | 2.12 | 2.11 | 0.06 | 7,665 | p95 |
| `ccm_E-10_mean` | ccm | causal | 136 | 1.15 | 1.54 | 0.14 | 1,346 | p99 |
| `ccm_E-10_max` | ccm | causal | 136 | 1.15 | 1.54 | 0.14 | 1,346 | p99 |
| `ccm_E-10_convergence` | ccm | causal | 136 | 1.15 | 1.54 | 0.14 | 1,346 | none |
| `ccm_E-None_max` | ccm | causal | 350 | 1.82 | 1.32 | 0.31 | 2,138 | p99 |
| `ccm_E-None_convergence` | ccm | causal | 350 | 1.82 | 1.32 | 0.31 | 2,138 | none |
| `ccm_E-None_mean` | ccm | causal | 350 | 1.82 | 1.32 | 0.31 | 2,138 | p99 |
| `hhg` | hhg | distance | 98.1 | 2.19 | 2.17 | 0.19 | 2,139 | none |
| `mgcx_maxlag-1` | mgcx | distance | 143 | 2.18 | 2.34 | 0.50 | 5,178 | p99 |

## Enters at full (3 SPIs)

| SPI | family | module | s M16 T1000 | p_M | q_T | rmse | proj s M16 T4000 | shipped |
|---|---|---|---:|---:|---:|---:|---:|---|
| `gpfit_DotProduct` | gpfit | misc | 679 | 2.07 | 2.31 | 0.16 | 18,591 | p99 |
| `gpfit_RBF` | gpfit | misc | 1,125 | 2.09 | 2.26 | 0.05 | 25,528 | none |
| `mgcx_maxlag-10` | mgcx | distance | 763 | 2.18 | 2.39 | 0.44 | 30,549 | none |
